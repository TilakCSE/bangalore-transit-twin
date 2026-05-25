// TransitDataBridge.cs
// ─────────────────────────────────────────────────────────────────────────────
// Manages the WebSocket connection to the FastAPI serving layer and drives
// all vehicle GameObject transforms in the Unity 3D scene.
//
// Architecture:
//   - One persistent WebSocket connection to /ws/vehicles
//   - Incoming JSON parsed on a background thread, queued to main thread
//   - Each vehicle is a pooled GameObject from VehiclePoolManager
//   - Position updates use smooth interpolation (no teleporting on GPS ping)
//   - Handles reconnect with exponential backoff
//   - Integrates with AlertOverlayManager for HUD delay/bunching indicators
//
// Usage:
//   Attach to an empty "TransitManager" GameObject in your scene.
//   Assign vehiclePool, alertManager, and cityOrigin in the Inspector.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using NativeWebSocket;           // from: https://github.com/endel/NativeWebSocket
using Newtonsoft.Json;            // from: com.unity.nuget.newtonsoft-json

namespace TransitTwin.DataBridge
{
    [Serializable]
    public class VehiclePositionData
    {
        public string vehicle_id;
        public string route_id;
        public string trip_id;
        public float  latitude;
        public float  longitude;
        public float  bearing;
        public float  speed_mps;
        public string current_status;
        public string feed;
        public long   timestamp;
        public int?   eta_seconds;
        public int?   delay_seconds;
        public bool   is_bunching;
    }

    [Serializable]
    public class VehicleSnapshot
    {
        public string type;    // "snapshot" | "update"
        public List<VehiclePositionData> data;
    }

    public class TransitDataBridge : MonoBehaviour
    {
        [Header("API Configuration")]
        [SerializeField] private string wsUrl = "ws://localhost:8000/ws/vehicles";
        [SerializeField] private string alertsWsUrl = "ws://localhost:8000/ws/alerts";
        [SerializeField] private float  reconnectDelayBase = 2f;
        [SerializeField] private int    maxReconnectAttempts = 10;

        [Header("Scene References")]
        [SerializeField] private VehiclePoolManager vehiclePool;
        [SerializeField] private AlertOverlayManager alertManager;
        [SerializeField] private Transform cityOriginTransform;

        [Header("City Geo-Reference (Bangalore centre)")]
        [SerializeField] private double originLatitude  = 12.9716;
        [SerializeField] private double originLongitude = 77.5946;
        [SerializeField] private float  metersPerDegreeLatitude  = 111320f;
        [SerializeField] private float  metersPerDegreeLongitude = 110540f;

        [Header("Interpolation")]
        [SerializeField] private float positionLerpSpeed = 4f;   // world units/sec
        [SerializeField] private float rotationLerpSpeed = 6f;

        // ── Internal state ─────────────────────────────────────────────────────
        private WebSocket _ws;
        private CancellationTokenSource _cts;
        private readonly ConcurrentQueue<VehiclePositionData> _updateQueue = new();
        private readonly Dictionary<string, VehicleController> _activeVehicles = new();
        private int _reconnectAttempts;

        // ── Unity lifecycle ────────────────────────────────────────────────────
        private void Start()
        {
            _cts = new CancellationTokenSource();
            StartCoroutine(ConnectWithBackoff());
        }

        private void Update()
        {
            // Drain the queue on the main thread (WebSocket messages arrive off-thread)
            while (_updateQueue.TryDequeue(out var data))
                ProcessVehicleUpdate(data);

            // Tick NativeWebSocket message dispatch
            _ws?.DispatchMessageQueue();
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
            _ = _ws?.Close();
        }

        // ── Connection management ──────────────────────────────────────────────
        private IEnumerator ConnectWithBackoff()
        {
            while (_reconnectAttempts < maxReconnectAttempts && !_cts.IsCancellationRequested)
            {
                Debug.Log($"[TransitBridge] Connecting to {wsUrl} (attempt {_reconnectAttempts + 1})");
                yield return ConnectAsync();

                _reconnectAttempts++;
                float delay = Mathf.Min(reconnectDelayBase * Mathf.Pow(2, _reconnectAttempts), 60f);
                Debug.LogWarning($"[TransitBridge] Disconnected. Reconnecting in {delay:F1}s");
                yield return new WaitForSeconds(delay);
            }
        }

        private IEnumerator ConnectAsync()
        {
            _ws = new WebSocket(wsUrl);

            _ws.OnOpen += () =>
            {
                Debug.Log("[TransitBridge] WebSocket connected");
                _reconnectAttempts = 0;
            };

            _ws.OnMessage += bytes =>
            {
                var json = System.Text.Encoding.UTF8.GetString(bytes);
                try
                {
                    var snapshot = JsonConvert.DeserializeObject<VehicleSnapshot>(json);
                    if (snapshot?.data == null) return;
                    foreach (var v in snapshot.data)
                        _updateQueue.Enqueue(v);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[TransitBridge] Parse error: {ex.Message}");
                }
            };

            _ws.OnError += err => Debug.LogError($"[TransitBridge] Error: {err}");
            _ws.OnClose += code => Debug.Log($"[TransitBridge] Closed: {code}");

            var connectTask = _ws.Connect();

            // Send keep-alive pings every 25 seconds
            StartCoroutine(SendPing());

            yield return new WaitUntil(() => connectTask.IsCompleted);
        }

        private IEnumerator SendPing()
        {
            while (_ws != null && _ws.State == WebSocketState.Open)
            {
                yield return new WaitForSeconds(25f);
                if (_ws.State == WebSocketState.Open)
                    _ = _ws.SendText("ping");
            }
        }

        // ── Vehicle update processing ──────────────────────────────────────────
        private void ProcessVehicleUpdate(VehiclePositionData data)
        {
            if (string.IsNullOrEmpty(data.vehicle_id)) return;

            var worldPos = GeoToWorld(data.latitude, data.longitude);

            if (_activeVehicles.TryGetValue(data.vehicle_id, out var controller))
            {
                // Update existing vehicle — smooth interpolation handled in VehicleController
                controller.SetTargetPose(worldPos, data.bearing);
                controller.UpdateData(data);
            }
            else
            {
                // Spawn new vehicle from pool
                var go = vehiclePool.Rent(data.feed, data.route_id);
                if (go == null) return;
                go.transform.position = worldPos;
                go.transform.rotation = Quaternion.Euler(0, data.bearing, 0);

                controller = go.GetComponent<VehicleController>();
                if (controller == null) controller = go.AddComponent<VehicleController>();
                controller.Initialize(data, positionLerpSpeed, rotationLerpSpeed);
                _activeVehicles[data.vehicle_id] = controller;
            }

            // Remove vehicles that haven't sent a position in >5 minutes
            CleanupStaleVehicles();
        }

        private Vector3 GeoToWorld(double lat, double lon)
        {
            // Equirectangular projection centred on Bangalore
            float x = (float)((lon - originLongitude) * metersPerDegreeLongitude);
            float z = (float)((lat - originLatitude)  * metersPerDegreeLatitude);
            return new Vector3(x, 0f, z);
        }

        private void CleanupStaleVehicles()
        {
            long nowSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var stale = new List<string>();
            foreach (var (id, ctrl) in _activeVehicles)
                if (nowSec - ctrl.LastSeenTimestamp > 300)
                    stale.Add(id);
            foreach (var id in stale)
            {
                vehiclePool.Return(_activeVehicles[id].gameObject);
                _activeVehicles.Remove(id);
            }
        }

        // ── Public API for DVR replay ──────────────────────────────────────────
        public void ReplayFrame(List<VehiclePositionData> frame)
        {
            foreach (var v in frame)
                _updateQueue.Enqueue(v);
        }
    }
}