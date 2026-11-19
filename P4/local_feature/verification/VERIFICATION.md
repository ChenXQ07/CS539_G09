# Verification on November 19, 2026

CPU execution on Windows, Python 3.12.10, torch 2.15.0+cpu, torchvision 0.29.2+cpu, pytest 9.2.0.

## Actual outcomes

- 18 tests passed in the original component run; all 18 passed again in 1.43 seconds after copying into the repository.
- Source-derived reference versus P4: identical float32 outputs, input gradients and parameter gradients (rtol=0, atol=0) in train and eval modes for the seeded test case.
- Both local convolution weights change after an optimizer step.
- Tested source diagram batch size 4 at 8x8,16x16,32x32; also rectangular 13x19 features.
- Verified BatchNorm train/eval running statistics, checkpoint round trip, invalid input rejection, and independent parameters across three instances.
- Alex custom encoder connected with weights=None and synthetic [2,3,256,256] input. Test-only 1x1 projections plus normalization prepare three local inputs. Outputs are [2,64,8,8], [2,64,16,16], [2,64,32,32]. Gradients reach encoder.conv1; local weights update. See encoder_connection.json.

## Not validated

Yin's actual global branch; complete shared decoder; real image-label batch; segmentation training/accuracy; GPU execution. No trained checkpoint is delivered. Adapters in the example are for connectivity only and do not implement the production decoder's skip fusion or feature refinement.

Alex's source package is preserved byte-for-byte. Reference source hash is in source_provenance.json. The encoder package's older verification claims are Alex's records, not checks rerun here.
