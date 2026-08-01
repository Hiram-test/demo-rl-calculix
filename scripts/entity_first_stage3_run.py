"""Correct CalculiX text parsing after preserving the first Stage 3 failure evidence."""  # Isolate the solver-adapter repair from deck and mesh logic.
from __future__ import annotations  # Enable modern annotations on Actions Python.
import math  # Validate parsed solver values.
import re  # Match exact CalculiX result-table rows.
import entity_first_stage3_calculix as stage3  # Reuse the verified mesh parser, deck generator, and solver suite.
NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"  # Define one CalculiX decimal or scientific number.
def corrected_build_deck(*args: object, **kwargs: object) -> str:  # Remove the unsupported ELSET parameter from the FRD output request.
    deck = ORIGINAL_BUILD_DECK(*args, **kwargs)  # Generate the complete deterministic deck with the original tested logic.
    return deck.replace("*EL FILE, ELSET=DOMAIN\n", "*EL FILE\n")  # Keep stress output while avoiding a documented solver warning.
def corrected_displacement_max(dat_text: str) -> float:  # Parse only displacement table values and exclude node IDs and time labels.
    section_match = re.search(r"displacements \(vx,vy,vz\).*?\n\s*\n(?P<body>.*?)(?:\n\s*\n\s*forces|\Z)", dat_text, flags=re.IGNORECASE | re.DOTALL)  # Capture the displacement rows up to the next result section.
    if section_match is None:  # Require the explicit requested displacement section.
        raise ValueError("CalculiX .dat file contains no displacement table")  # Reject missing text-result evidence.
    row_pattern = re.compile(rf"^\s*\d+\s+(?P<ux>{NUMBER})\s+(?P<uy>{NUMBER})\s+(?P<uz>{NUMBER})\s*$", flags=re.MULTILINE)  # Match one node ID followed by three displacement components.
    magnitudes: list[float] = []  # Collect absolute displacement components only.
    for row in row_pattern.finditer(section_match.group("body")):  # Parse every displacement row in the captured table.
        components = (float(row.group("ux")), float(row.group("uy")), float(row.group("uz")))  # Convert the three physical components.
        if not all(math.isfinite(value) for value in components):  # Reject any non-finite solver output.
            raise ValueError("CalculiX displacement table contains non-finite values")  # Preserve result-integrity failure.
        magnitudes.extend(abs(value) for value in components)  # Store component magnitudes without node identifiers.
    if not magnitudes:  # Require at least one parsed displacement row.
        raise ValueError("CalculiX displacement table contains no result rows")  # Reject a header-only result section.
    return max(magnitudes)  # Return the largest absolute physical displacement component.
def corrected_reaction_sum_y(dat_text: str) -> float:  # Parse the solver-computed total reaction vector from its following data row.
    total_match = re.search(rf"total force \(fx,fy,fz\).*?\n\s*\n\s*(?P<fx>{NUMBER})\s+(?P<fy>{NUMBER})\s+(?P<fz>{NUMBER})", dat_text, flags=re.IGNORECASE | re.DOTALL)  # Match the exact three-component total-force row.
    if total_match is None:  # Require support equilibrium evidence.
        raise ValueError("CalculiX .dat file contains no total reaction vector")  # Reject incomplete or differently requested output.
    values = (float(total_match.group("fx")), float(total_match.group("fy")), float(total_match.group("fz")))  # Convert the physical reaction components.
    if not all(math.isfinite(value) for value in values):  # Reject non-finite equilibrium output.
        raise ValueError("CalculiX total reaction vector contains non-finite values")  # Preserve the exact failure class.
    return values[1]  # Return the vertical reaction component.
ORIGINAL_BUILD_DECK = stage3.build_deck  # Preserve the original generator before installing the compatibility repair.
stage3.build_deck = corrected_build_deck  # Apply only the unsupported-output-card correction.
stage3.parse_displacement_max = corrected_displacement_max  # Apply the exact displacement-table parser.
stage3.parse_reaction_sum_y = corrected_reaction_sum_y  # Apply the exact total-reaction parser.
if __name__ == "__main__":  # Run only when invoked directly by the isolated Stage 3 workflow.
    raise SystemExit(stage3.main())  # Execute the unchanged solver suite through the corrected adapters.