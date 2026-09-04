# GPT visual AMR evidence reconstruction

Full archive SHA-256: `a07903dc28022b92982c116417de3dbe622ae7acdc0ccdfebeb1afeac05fc949`

Full archive bytes: 20907631

Transport part order:

1. `gpt_visual_evidence.zip.part001`
2. `gpt_visual_evidence.zip.part002`
3. `gpt_visual_evidence.zip.part003`
4. `gpt_visual_evidence.zip.part004`
5. `gpt_visual_evidence.zip.part005`
6. `gpt_visual_evidence.zip.part006`
7. `gpt_visual_evidence.zip.part007`
8. `gpt_visual_evidence.zip.part008`
9. `gpt_visual_evidence.zip.part009`
10. `gpt_visual_evidence.zip.part010`
11. `gpt_visual_evidence.zip.part011`
12. `gpt_visual_evidence.zip.part012`
13. `gpt_visual_evidence.zip.part013`
14. `gpt_visual_evidence.zip.part014`

Run in the directory containing the downloaded parts and checksum receipts:

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt # Verify the downloaded transport pieces and receipts.
cat gpt_visual_evidence.zip.part[0-9][0-9][0-9] > gpt_visual_evidence.zip # Reconstruct the exact complete archive in numeric part order.
sha256sum -c SHA256SUMS.txt # Verify all downloaded pieces and the complete reconstructed ZIP.
unzip -t gpt_visual_evidence.zip # Check the ZIP's internal CRC integrity.
unzip gpt_visual_evidence.zip -d gpt_visual_evidence # Extract the preserved relative source and evidence paths.
(cd gpt_visual_evidence && sha256sum -c CONTENT_SHA256SUMS.txt) # Verify every archived scientific and source artifact.
```

Final-field post/reference NPZ containers preserve exact original nodes, cells, and displacement NPY bytes; their compact archive hashes differ from their recorded original complete NPZ hashes. Initial and observation NPZ bytes remain unchanged. See EVIDENCE_MANIFEST.json for the explicit verification limit.
