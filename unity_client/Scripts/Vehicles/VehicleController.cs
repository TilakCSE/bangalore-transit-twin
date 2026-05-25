// VehicleController.cs
// ─────────────────────────────────────────────────────────────────────────────
// Attached to each vehicle GameObject in the digital twin.
// Handles smooth position + rotation interpolation between GPS updates,
// visual state changes (delayed, bunching, on-time), and ETA label display.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using TMPro;
using TransitTwin.DataBridge;
using UnityEngine;

namespace TransitTwin.Vehicles
{
    public class VehicleController : MonoBehaviour
    {
        [Header("Visual Components")]
        [SerializeField] private Renderer bodyRenderer;
        [SerializeField] private TMP_Text labelText;
        [SerializeField] private GameObject delayIndicator;
        [SerializeField] private GameObject bunchingIndicator;
        [SerializeField] private TrailRenderer speedTrail;

        [Header("Materials")]
        [SerializeField] private Material matOnTime;
        [SerializeField] private Material matDelayed;
        [SerializeField] private Material matEarly;
        [SerializeField] private Material matBunching;

        // ── State ──────────────────────────────────────────────────────────────
        private Vector3    _targetPosition;
        private Quaternion _targetRotation;
        private float      _posLerpSpeed;
        private float      _rotLerpSpeed;
        private VehiclePositionData _data;

        public long LastSeenTimestamp { get; private set; }

        // ── Init ───────────────────────────────────────────────────────────────
        public void Initialize(VehiclePositionData data, float posSpeed, float rotSpeed)
        {
            _posLerpSpeed = posSpeed;
            _rotLerpSpeed = rotSpeed;
            _targetPosition = transform.position;
            _targetRotation = transform.rotation;
            UpdateData(data);
        }

        // ── Called by TransitDataBridge every time a new position arrives ──────
        public void SetTargetPose(Vector3 worldPos, float bearing)
        {
            _targetPosition = worldPos;
            _targetRotation = Quaternion.Euler(0f, bearing, 0f);
        }

        public void UpdateData(VehiclePositionData data)
        {
            _data = data;
            LastSeenTimestamp = data.timestamp;
            RefreshVisuals();
        }

        // ── Unity Update: smooth interpolation ────────────────────────────────
        private void Update()
        {
            transform.position = Vector3.Lerp(
                transform.position, _targetPosition, Time.deltaTime * _posLerpSpeed
            );
            transform.rotation = Quaternion.Slerp(
                transform.rotation, _targetRotation, Time.deltaTime * _rotLerpSpeed
            );

            // Adjust trail length based on speed (visual feedback)
            if (speedTrail != null)
                speedTrail.time = Mathf.Clamp(_data?.speed_mps ?? 0f / 15f, 0.1f, 1.5f);
        }

        // ── Visuals ────────────────────────────────────────────────────────────
        private void RefreshVisuals()
        {
            if (_data == null) return;

            // Material state
            if (bodyRenderer != null)
            {
                Material mat = matOnTime;
                if (_data.is_bunching && matBunching != null)
                    mat = matBunching;
                else if (_data.delay_seconds.HasValue)
                {
                    if (_data.delay_seconds.Value > 120)
                        mat = matDelayed;
                    else if (_data.delay_seconds.Value < -60)
                        mat = matEarly;
                }
                bodyRenderer.material = mat;
            }

            // Delay / bunching indicator icons
            delayIndicator?.SetActive(_data.delay_seconds.HasValue && _data.delay_seconds.Value > 120);
            bunchingIndicator?.SetActive(_data.is_bunching);

            // ETA label
            if (labelText != null)
            {
                if (_data.eta_seconds.HasValue)
                {
                    int min = _data.eta_seconds.Value / 60;
                    labelText.text = min <= 0 ? "NOW" : $"{min}m";
                    labelText.color = _data.delay_seconds > 120 ? Color.red
                                    : _data.delay_seconds < -60  ? Color.cyan
                                    : Color.green;
                }
                else
                {
                    labelText.text = _data.route_id ?? "";
                }
            }
        }

        // ── Pool lifecycle ─────────────────────────────────────────────────────
        public void OnRented()
        {
            gameObject.SetActive(true);
            if (speedTrail != null) speedTrail.Clear();
        }

        public void OnReturned()
        {
            gameObject.SetActive(false);
            _data = null;
        }
    }
}