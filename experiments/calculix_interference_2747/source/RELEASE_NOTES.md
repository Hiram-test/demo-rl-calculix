# CalculiX Interference Contact 2747 — Original Models and Evidence v1

This prerelease preserves the original large CalculiX inputs shared for the public forum question [Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747).

## Assets

- `Shear_setups.zip`: exact author-published archive.
- `SOURCE_MANIFEST.json`: source, role, size, hash and execution-boundary manifest.
- `SHA256SUMS.txt`: download verification.
- `SOURCE_AND_REDISTRIBUTION_NOTICE.md`: third-party source and rights boundary.
- `local_calculix_counterfactual_followup.pdf`: local reduced-model counterfactual report.

The original ZIP contains exactly two complete, self-contained inputs: `Shear_setup-INTER01-COARSE_PIN.inp` and `Shear_setup-INTER01-deactivate1thenreactivate.inp`. Their independent hashes are included in the manifest and `SHA256SUMS.txt`. They are not duplicated as separate 267 MB and 277 MB assets because extraction preserves the exact entry bytes.

## Evidence boundary

The two original models contain approximately 3.84–3.86 million nodes and request Pardiso. They have **not** been executed by this repository on the current workstation or standard GitHub runner. The attached report uses real CalculiX 2.22/SPOOLES reduced models and narrows possible mechanisms; it does not establish a confirmed fix for either original input.

The current engineering state is `narrowed_unresolved`, so this is a prerelease rather than a stable solution release.

## Third-party notice

The original archive contains no explicit license. This repository does not claim ownership and does not grant rights over the original models. The mirror preserves public research evidence with source attribution and exact hashes; users must determine the rights applicable to their own use. See `SOURCE_AND_REDISTRIBUTION_NOTICE.md`.

The three separate research-user challenges—phase-field fracture, classic-to-nonclassic transfer, and composition of established methods—are reviewable text artifacts in [PR #20](https://github.com/Hiram-test/demo-rl-calculix/pull/20), not silently bundled into this contact-model release.
