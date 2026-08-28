# Instrument facts for the paper's setup paragraph

Source: the mentor (S.), email of 2026-08-28. Quote as given; nothing below
is inferred.

- Two Amptek FAST SDD silicon drift detectors
  (https://www.amptek.com/products/x-ray-detectors/fastsdd-x-ray-detectors-for-xrf/fastsdd-silicon-drift-detector).
- Detector 19511: mounted horizontally, 70 mm^2 active area.
- Detector 10264: mounted 40 degrees from horizontal, 25 mm^2 active area.
- Both detectors: 12.5 um beryllium window.
- X-ray tube: 37 kV, 40 uA.

Relation to the measured forward model: the maps used everywhere in this
package are the detector-summed net counts (10264 + 19511). The per-line
tilt gains of `forward_model.tilt_gains` (script 11, column tilt_pct_sum)
are the summed response; the two detectors respond to the tilt with
opposite signs per degree (script 11, per-detector columns), consistent
with their different take-off geometry above. No solid-angle or
distance modelling is done in this package.
