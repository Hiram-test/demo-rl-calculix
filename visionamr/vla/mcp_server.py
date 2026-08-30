"""Optional MCP v2 server exposing deterministic mesh-parameter tools."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from mcp.server import MCPServer  # Import the official MCP Python SDK v2 server class.
from ..bridge_cases import make_box_girder_diaphragm  # Import the canonical bridge case without running a solve.
from .mcp_tools import regional_level_targets  # Reuse the exact local parameter implementation.
mcp = MCPServer("visionamr-mesh-tools", instructions="Use these tools for exact mesh parameters; do not invent continuous sizes.")  # Create the typed MCP tool server.

@mcp.tool()  # Expose case inspection through the standard tool protocol.
def inspect_box_girder_case() -> dict:  # Return immutable geometry, units, and named mechanisms.
    """Return the canonical three-dimensional bridge-component definition."""  # Supply MCP tool metadata.
    problem = make_box_girder_diaphragm()  # Build only the Python problem definition.
    return {"name": problem.name, "instance_id": problem.instance_id, "units": {"length": "mm", "stress": "MPa", "force": "N"}, "dimension": problem.dim, "parameters": problem.params, "mesh_bounds": {"h_min": problem.h_min, "h0": problem.h0, "h_ref": problem.h_ref}, "features": [{"name": feature.name, "kind": feature.kind, "xyz": [feature.x, feature.y, feature.z]} for feature in problem.features], "constraints": [{"name": item.name, "dofs": list(item.dofs)} for item in problem.constraints], "tractions": [{"name": item.name, "value": list(item.value)} for item in problem.tractions]}  # Return exact repository-owned values.

@mcp.tool()  # Expose discrete-to-continuous size conversion as a typed tool.
def materialize_region_levels(current_sizes: list[float], extra_depth: list[int], refine_factor: float, h_min: float, h_max: float) -> dict:  # Convert validated integer levels into exact bounded sizes.
    """Convert regional integer depths to exact sizes with no model-generated numbers."""  # Supply MCP tool metadata.
    targets = regional_level_targets(current_sizes, extra_depth, refine_factor, h_min, h_max)  # Execute the shared deterministic conversion.
    return {"current_sizes": current_sizes, "extra_depth": extra_depth, "refine_factor": refine_factor, "h_min": h_min, "h_max": h_max, "target_sizes": targets}  # Return an auditable conversion record.

@mcp.tool()  # Expose action validation as a typed tool.
def validate_region_action(region_names: list[str], extra_depth: list[int], predicted_equations: float, member_equations: list[float], equation_budget: int, max_extra_regions: int = 2, max_extra_depth: int = 2, safety: float = 0.98) -> dict:  # Validate sparse depths and ensemble resource bounds.
    """Validate action dimensions, depth limits, sparsity, and the equation cap."""  # Supply MCP tool metadata.
    reasons: list[str] = []  # Allocate the full validation report.
    if len(region_names) != len(extra_depth):  # Require one depth per named region.
        reasons.append("dimension_mismatch")  # Record the structural mismatch.
    if any(int(value) != value or value < 0 for value in extra_depth):  # Require non-negative integer depths.
        reasons.append("invalid_depth_value")  # Record an invalid action entry.
    if sum(value > 0 for value in extra_depth) > max_extra_regions:  # Enforce sparse proactive investment.
        reasons.append("too_many_proactive_regions")  # Record the sparsity violation.
    if max(extra_depth, default=0) > max_extra_depth:  # Enforce the configured depth cap.
        reasons.append("depth_limit_exceeded")  # Record the depth violation.
    if equation_budget <= 0:  # Require a positive resource cap.
        reasons.append("invalid_budget")  # Record the invalid budget.
    upper_equations = max(member_equations, default=predicted_equations)  # Use the most conservative ensemble resource prediction.
    if predicted_equations <= 0.0 or upper_equations <= 0.0:  # Require positive resource predictions.
        reasons.append("invalid_equation_prediction")  # Record an invalid model result.
    if equation_budget > 0 and upper_equations > safety * equation_budget and any(value > 0 for value in extra_depth):  # Reserve budget headroom for proactive actions.
        reasons.append("proactive_budget_margin_failed")  # Record the resource-margin failure.
    return {"accepted": not reasons, "reasons": reasons, "region_names": region_names, "extra_depth": extra_depth, "predicted_equations": predicted_equations, "upper_member_equations": upper_equations, "equation_budget": equation_budget, "safety": safety}  # Return the complete validation record.

@mcp.tool()  # Expose the Dörfler dominance check as a typed tool.
def certify_dorfler_dominance(dorfler_targets: list[float], candidate_targets: list[float], tolerance: float = 1.0e-12) -> dict:  # Check that no candidate node is coarser than Dörfler.
    """Certify nodewise target-size dominance relative to pure Dörfler."""  # Supply MCP tool metadata.
    reasons: list[str] = []  # Allocate the dominance report.
    if len(dorfler_targets) != len(candidate_targets):  # Require nodal vector alignment.
        reasons.append("dimension_mismatch")  # Record the structural mismatch.
    else:  # Compare aligned nodal targets.
        violating = [index for index, (baseline, candidate) in enumerate(zip(dorfler_targets, candidate_targets)) if candidate > baseline + tolerance]  # Locate every coarser candidate target.
        if violating:  # Reject any local coarsening relative to Dörfler.
            reasons.append("candidate_is_coarser_than_dorfler")  # Record the monotonicity violation.
    return {"accepted": not reasons, "reasons": reasons, "n_nodes": len(candidate_targets), "tolerance": tolerance}  # Return the exact dominance certificate.

if __name__ == "__main__":  # Run only when launched as an MCP server process.
    mcp.run(transport="stdio")  # Serve typed tools over the standard input-output transport.
