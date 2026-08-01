"""Run one clean DeepSeek engineering session that configures and reviews one hotspot PSO case."""  # Keep every case in an independent process and message history.
from __future__ import annotations  # Enable modern annotations on the GitHub Actions Python runtime.
from dataclasses import asdict, replace  # Serialize evidence and create validated case overrides.
from hashlib import sha256  # Seal the complete local session history for audit.
from pathlib import Path  # Resolve exact-entity inputs and case-isolated outputs safely.
from typing import Any, Sequence  # Declare JSON and numerical-vector contracts explicitly.
import argparse  # Parse one case, entity root, output root, and PSO depth.
import json  # Encode API requests and persist complete session evidence.
import os  # Read the DeepSeek credential and GitHub run identity.
import re  # Recover a JSON object when a provider wraps it in a code fence.
import sys  # Import the deterministic Stage 5 optimizer from the repository.
import time  # Record wall-clock evidence and bounded API retry delays.
import urllib.error  # Preserve HTTP failures without hiding provider evidence.
import urllib.request  # Call the official OpenAI-compatible DeepSeek endpoint with no extra SDK.

ROOT = Path(__file__).resolve().parents[1]  # Resolve the repository root from this script.
SCRIPTS = ROOT / "scripts"  # Resolve sibling deterministic pipeline modules.
if str(SCRIPTS) not in sys.path:  # Ensure direct execution imports the checked-out repository code.
    sys.path.insert(0, str(SCRIPTS))  # Put repository scripts before ambient packages.

import entity_first_stage5_hotspot_pso as stage5  # Reuse exact Gmsh fields, seam creation, audits, and PSO.

CASE_FACTS: dict[str, dict[str, Any]] = {  # Freeze the public model facts visible to each isolated session.
    "bearing_plate": {  # Describe the intact rectangular bearing/load-path plate.
        "question": "在固定实体不变的条件下，怎样把网格资源明显集中到固定端和有限宽度载荷传递端，同时释放中跨网格？",  # Give the engineering task in natural language.
        "model": "1000 mm × 100 mm完整矩形板；左边界固定，右边界为有限宽度载荷边；无孔洞。",  # Prevent the model from inventing geometry.
        "semantics": "两个热点分别是固定端全高载荷路径区和右端载荷传递区，远离两端的中跨为可粗化背景。",  # Define the tested hotspot family.
    },  # Complete the bearing facts.
    "circular_opening": {  # Describe the exact single-opening BREP.
        "question": "在圆孔几何严格不变的条件下，怎样形成孔顶和孔底明显加密、远场明显粗化的网格？",  # Give the engineering task in natural language.
        "model": "240 mm × 240 mm完整板减去中心(120,120) mm、半径20 mm的精确圆孔。",  # Preserve exact hole geometry.
        "semantics": "热点只针对圆孔顶部和底部的局部响应区；圆孔几何、边界环和材料域不能被PSO改变。",  # Define the tested hotspot family.
    },  # Complete the single-hole facts.
    "three_openings": {  # Describe the exact three-opening BREP.
        "question": "三个孔竞争有限网格资源时，怎样让中孔最密、左孔次之、右孔相对较疏，并与远场形成明显区分？",  # Give the engineering task in natural language.
        "model": "600 mm × 260 mm完整板，三个精确圆孔分别为(130,130,r24)、(300,130,r42)、(450,130,r30) mm。",  # Preserve all exact holes.
        "semantics": "每个孔的顶部和底部热点由独立尺寸与范围控制；目标次序固定为中孔最密、左孔次之、右孔最疏。",  # Require a genuine multi-hotspot allocation.
    },  # Complete the three-hole facts.
    "cracked_web": {  # Describe the intact crack-panel entity and seam representation.
        "question": "在完整板实体不挖孔的条件下，怎样把网格集中到两个裂尖和裂纹尾迹，并保持远场明显粗化？",  # Give the fracture hotspot task.
        "model": "200 mm × 200 mm完整板；裂纹导向线从(70,100)到(130,100) mm；网格后沿内部线复制重合节点形成零厚度seam。",  # Keep crack distinct from material removal.
        "semantics": "两个裂尖共享一个局部尺寸变量，裂纹尾迹独立控制；上下裂纹面节点坐标重合但编号和自由度分离。",  # Define the seam-specific hotspot family.
    },  # Complete the crack facts.
}  # Finish public case facts.

SAFE_ENVELOPES: dict[str, dict[str, Any]] = {  # Restrict DeepSeek proposals to already tested deterministic domains.
    case_id: {  # Derive each envelope from the validated Stage 5 case definition.
        "names": list(case.names),  # Preserve parameter ordering exactly.
        "minimum": list(case.lower),  # Set the lowest permitted lower bound.
        "maximum": list(case.upper),  # Set the highest permitted upper bound.
        "target_ratio_count": len(case.target_ratios),  # Require the model-specific number of density ratios.
        "target_elements_min": max(400, int(case.target_elements * 0.55)),  # Permit meaningful but bounded budget changes.
        "target_elements_max": int(case.target_elements * 1.80),  # Prevent an unbounded element-count request.
    }  # Complete one safe proposal envelope.
    for case_id, case in stage5.CASES.items()  # Cover all four validated hotspot cases.
}  # Finish deterministic envelopes.


def canonical_json(value: Any) -> str:  # Encode stable JSON for hashing and audit comparisons.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # Remove irrelevant whitespace.


def extract_json_object(text: str) -> dict[str, Any]:  # Parse one provider response as a JSON object.
    stripped = text.strip()  # Remove surrounding whitespace first.
    if stripped.startswith("```"):  # Detect a Markdown code fence despite the JSON-only instruction.
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)  # Remove the opening fence.
        stripped = re.sub(r"\s*```$", "", stripped)  # Remove the closing fence.
    try:  # Attempt direct strict JSON parsing.
        parsed = json.loads(stripped)  # Decode the complete response.
    except json.JSONDecodeError:  # Fall back only to the outermost object span.
        start = stripped.find("{")  # Locate the first object delimiter.
        end = stripped.rfind("}")  # Locate the last object delimiter.
        if start < 0 or end <= start:  # Reject prose without a complete object.
            raise ValueError("DeepSeek response contains no complete JSON object")  # Preserve a clear failure class.
        parsed = json.loads(stripped[start : end + 1])  # Decode only the complete object span.
    if not isinstance(parsed, dict):  # Require named fields rather than a scalar or list.
        raise ValueError("DeepSeek response must be a JSON object")  # Reject an incompatible response shape.
    return parsed  # Return the decoded proposal or review.


class DeepSeekSession:  # Maintain one case-local message history and API trace.
    def __init__(self, case_id: str, output_root: Path) -> None:  # Create a brand-new session for exactly one case.
        self.case_id = case_id  # Preserve the only case allowed in this context.
        self.output_root = output_root  # Preserve the case-isolated evidence directory.
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # Read the masked environment credential.
        if not self.api_key:  # Fail before any numerical work when the live credential is missing.
            raise RuntimeError("DEEPSEEK_API_KEY is empty")  # Avoid silently falling back to deterministic text.
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # Read the configured live model.
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")  # Fix the API root.
        self.max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "5000"))  # Bound each response length.
        run_id = os.environ.get("GITHUB_RUN_ID", "local")  # Read the GitHub run identity when available.
        run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")  # Read the rerun attempt identity.
        self.session_id = f"github-{run_id}-attempt-{run_attempt}-{case_id}"  # Make cross-case sharing impossible by construction.
        self.messages: list[dict[str, str]] = []  # Start with an empty local conversation history.
        self.calls: list[dict[str, Any]] = []  # Preserve request/response and usage evidence for every model call.
        self.started_from_empty_history = len(self.messages) == 0  # Record the requested clean-session invariant.

    def call_json(self, user_content: str, purpose: str) -> dict[str, Any]:  # Send one turn and require a JSON object.
        if not self.messages:  # Add the system contract only on the first call of this case.
            self.messages.append({  # Create the isolated system message.
                "role": "system",  # Mark the instruction as the session authority.
                "content": (  # Define model scope and numerical authority.
                    "你是有限元区域网格配置分析员。你只处理当前一个案例，不得引用其他案例的会话、结论或隐藏历史。"
                    "几何、边界拓扑、孔洞和裂纹seam由确定性Gmsh/CalculiX流程控制；你只能提出热点PSO的参数边界、目标密度比例、单元预算和首个粒子。"
                    "不得改变实体，不得把裂纹当孔，不得声称已经求解。所有回复必须是单个JSON对象，不要Markdown。"
                ),  # Finish the isolated system contract.
            })  # Complete the first system message.
        prior_message_count = len(self.messages)  # Record context length before adding this user turn.
        self.messages.append({"role": "user", "content": user_content})  # Add only this case's current evidence.
        payload = {  # Build the OpenAI-compatible chat-completions request.
            "model": self.model,  # Use the workflow-configured DeepSeek model.
            "messages": self.messages,  # Send only this case-local history.
            "temperature": 0.15,  # Keep numerical proposals stable while retaining independent reasoning.
            "max_tokens": self.max_tokens,  # Enforce the configured response cap.
            "response_format": {"type": "json_object"},  # Request strict JSON mode when supported.
        }  # Complete the provider request.
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # Encode the request body once.
        request = urllib.request.Request(  # Construct the authenticated HTTPS request.
            f"{self.base_url}/chat/completions",  # Use the official compatible endpoint.
            data=encoded,  # Attach the JSON body.
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},  # Supply masked auth and media type.
            method="POST",  # Use the required HTTP method.
        )  # Complete the request object.
        started = time.monotonic()  # Start per-call wall-clock timing.
        last_error: Exception | None = None  # Preserve the final bounded retry failure.
        for attempt in range(1, 3):  # Permit one retry for a transient provider failure.
            try:  # Execute the live request.
                with urllib.request.urlopen(request, timeout=240) as response:  # Bound one live call to four minutes.
                    raw_body = response.read().decode("utf-8")  # Read the complete provider response.
                response_object = json.loads(raw_body)  # Decode the API envelope.
                choice = response_object["choices"][0]  # Select the only requested completion.
                content = str(choice["message"]["content"])  # Read the assistant JSON text.
                parsed = extract_json_object(content)  # Require a valid JSON object before continuing.
                self.messages.append({"role": "assistant", "content": content})  # Extend only this session's history.
                self.calls.append({  # Preserve complete non-secret call evidence.
                    "purpose": purpose,  # Record why this call occurred.
                    "attempt": attempt,  # Record transient retries.
                    "prior_message_count": prior_message_count,  # Prove the first call began with no inherited history.
                    "sent_message_count": len(payload["messages"]),  # Record actual context size.
                    "request_case_id": self.case_id,  # Bind the request to one case.
                    "response": parsed,  # Preserve the parsed model output.
                    "usage": response_object.get("usage", {}),  # Preserve provider token accounting.
                    "finish_reason": choice.get("finish_reason"),  # Preserve truncation or stop evidence.
                    "elapsed_seconds": time.monotonic() - started,  # Preserve live wall-clock time.
                })  # Complete one call record.
                self._write_trace(status="running")  # Persist evidence after every successful call.
                return parsed  # Return the structured model output.
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as error:  # Capture network, envelope, and JSON failures.
                last_error = error  # Preserve the latest failure.
                self.calls.append({  # Write the failed attempt into the trace.
                    "purpose": purpose,  # Preserve call purpose.
                    "attempt": attempt,  # Preserve retry index.
                    "prior_message_count": prior_message_count,  # Preserve context size before this turn.
                    "request_case_id": self.case_id,  # Bind the failure to one case.
                    "error": f"{type(error).__name__}: {error}",  # Preserve a non-secret error description.
                    "elapsed_seconds": time.monotonic() - started,  # Preserve time spent before failure.
                })  # Complete the failed call record.
                self._write_trace(status="provider_retry" if attempt == 1 else "provider_failed")  # Persist failure evidence immediately.
                if attempt == 1:  # Delay only before the single permitted retry.
                    time.sleep(3.0)  # Use a short deterministic retry delay.
        raise RuntimeError(f"DeepSeek call failed after two attempts: {last_error}")  # Stop rather than fabricate a proposal.

    def _write_trace(self, status: str) -> None:  # Persist current session state without exposing credentials.
        trace = {  # Build the machine-readable session trace.
            "schema_version": "entity-first-live-deepseek-session/1.0",  # Version the isolation contract.
            "status": status,  # Record current execution state.
            "case_id": self.case_id,  # Record the only case in this session.
            "session_id": self.session_id,  # Record the unique local conversation identity.
            "started_from_empty_history": self.started_from_empty_history,  # Prove no inherited messages were loaded.
            "model": self.model,  # Record the requested live model.
            "messages": self.messages,  # Preserve the complete case-local conversation.
            "calls": self.calls,  # Preserve per-call request and usage evidence.
            "messages_sha256": sha256(canonical_json(self.messages).encode("utf-8")).hexdigest(),  # Seal the current history.
        }  # Complete the trace object.
        self.output_root.mkdir(parents=True, exist_ok=True)  # Ensure the isolated evidence directory exists.
        (self.output_root / "deepseek_session_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")  # Write the trace atomically enough for CI evidence.


def proposal_prompt(case_id: str) -> str:  # Build the first case-specific analysis request.
    facts = CASE_FACTS[case_id]  # Read only this case's public facts.
    envelope = SAFE_ENVELOPES[case_id]  # Read deterministic parameter limits.
    return (  # Return a strict JSON proposal request.
        f"当前案例ID：{case_id}\n工程问题：{facts['question']}\n模型事实：{facts['model']}\n热点语义：{facts['semantics']}\n"
        f"PSO参数顺序：{json.dumps(envelope['names'], ensure_ascii=False)}\n"
        f"每维允许总范围下限：{json.dumps(envelope['minimum'])}\n每维允许总范围上限：{json.dumps(envelope['maximum'])}\n"
        f"目标密度比例数量：{envelope['target_ratio_count']}；目标单元数允许范围：[{envelope['target_elements_min']},{envelope['target_elements_max']}]。\n"
        "请独立分析并返回：analysis_summary字符串、hotspots数组、lower数组、upper数组、target_ratios数组、target_elements整数、initial_position数组、engineering_boundary字符串。"
        "lower/upper/initial_position必须与参数顺序等长；每维minimum<=lower<upper<=maximum；initial_position必须位于lower与upper之间；所有target_ratios必须在0.06到0.55之间。"
    )  # Complete the first proposal prompt.


def revision_prompt(case_id: str, preview: dict[str, Any], previous: dict[str, Any]) -> str:  # Ask the same session to revise after a real Gmsh probe.
    return (  # Return only this case's numerical observation.
        f"仍然只处理案例{case_id}。你上一轮方案已在固定实体上执行一次真实Gmsh试网格。"
        f"上一轮结构化方案：{canonical_json(previous)}。"
        f"试网格证据：{canonical_json(preview)}。"
        "请根据实际单元数、最小角、最大半径比、局部/远场中位尺寸比例和任何issues修订方案。"
        "返回与上一轮完全相同的JSON字段；不得改变实体或热点语义；三孔必须保持中孔目标比例<左孔目标比例<右孔目标比例；裂纹必须保持裂尖与尾迹两个目标比例。"
    )  # Complete the revision prompt.


def final_review_prompt(case_id: str, best: dict[str, Any], history: Sequence[dict[str, Any]]) -> str:  # Ask the same session to review deterministic final evidence.
    compact_history = [  # Reduce convergence evidence to one row per generation.
        {"iteration": row["iteration"], "global_best_score": row["global_best_score"]}  # Preserve only the relevant convergence fields.
        for row in history  # Visit all completed PSO generations.
    ]  # Complete the compact convergence history.
    return (  # Return a strict final-review request.
        f"案例{case_id}的确定性热点PSO已经完成。最终best证据：{canonical_json(best)}。"
        f"每轮全局最优：{canonical_json(compact_history)}。"
        "请返回JSON对象，字段必须为accept_hotspot_partition布尔值、engineering_summary字符串、concerns字符串数组、next_action字符串、usage_boundary字符串。"
        "只评价热点分区清晰度、实体/裂纹拓扑、网格质量和搜索趋势，不得声称有限元误差已经优化。"
    )  # Complete the final review prompt.


def float_vector(value: Any, field: str, expected: int) -> tuple[float, ...]:  # Normalize one required numerical vector.
    if not isinstance(value, list) or len(value) != expected:  # Enforce exact dimensionality.
        raise ValueError(f"{field} must be a list of length {expected}")  # Reject missing or extra dimensions.
    result = tuple(float(item) for item in value)  # Convert every value to a finite float.
    if not all(item == item and abs(item) < 1.0e9 for item in result):  # Reject NaN and implausible overflow.
        raise ValueError(f"{field} contains a non-finite value")  # Preserve a clear validation error.
    return result  # Return the normalized vector.


def validate_proposal(case_id: str, proposal: dict[str, Any]) -> tuple[stage5.Case, tuple[float, ...]]:  # Convert a DeepSeek proposal into a safe deterministic case.
    base = stage5.CASES[case_id]  # Read the already validated base model and parameter names.
    envelope = SAFE_ENVELOPES[case_id]  # Read safe numerical limits.
    lower = float_vector(proposal.get("lower"), "lower", len(base.names))  # Normalize proposed lower bounds.
    upper = float_vector(proposal.get("upper"), "upper", len(base.names))  # Normalize proposed upper bounds.
    initial = float_vector(proposal.get("initial_position"), "initial_position", len(base.names))  # Normalize the proposed seed.
    ratios = float_vector(proposal.get("target_ratios"), "target_ratios", len(base.target_ratios))  # Normalize target density ratios.
    for index, (candidate_lower, candidate_upper, safe_lower, safe_upper, seed_value) in enumerate(zip(lower, upper, envelope["minimum"], envelope["maximum"], initial)):  # Validate every PSO dimension.
        if not (safe_lower <= candidate_lower < candidate_upper <= safe_upper):  # Enforce the pre-tested deterministic domain.
            raise ValueError(f"dimension {index} violates safe envelope [{safe_lower},{safe_upper}]")  # Reject silent clamping.
        if not (candidate_lower <= seed_value <= candidate_upper):  # Require the model-proposed first particle to be feasible.
            raise ValueError(f"initial_position[{index}] lies outside proposed bounds")  # Reject an unusable seed.
    if not all(0.06 <= ratio <= 0.55 for ratio in ratios):  # Enforce a clearly refined but nonzero target ratio.
        raise ValueError("target_ratios must stay in [0.06,0.55]")  # Reject a weak or impossible density target.
    if case_id == "three_openings" and not (ratios[1] < ratios[0] < ratios[2]):  # Preserve the requested independent three-hole order.
        raise ValueError("three_openings target ratios must satisfy middle < left < right")  # Reject a collapsed repeated-hole proposal.
    target_elements = int(proposal.get("target_elements"))  # Read the requested element budget.
    if not (envelope["target_elements_min"] <= target_elements <= envelope["target_elements_max"]):  # Enforce a realistic budget range.
        raise ValueError("target_elements is outside the safe case envelope")  # Reject an unachievable budget silently dominating PSO.
    configured = replace(base, lower=lower, upper=upper, target_elements=target_elements, target_ratios=ratios)  # Preserve geometry while applying only optimizer configuration.
    return configured, initial  # Return the safe deterministic case and DeepSeek seed.


def proposal_with_correction(session: DeepSeekSession, prompt: str, purpose: str) -> tuple[dict[str, Any], stage5.Case, tuple[float, ...]]:  # Give one case-local correction opportunity.
    current_prompt = prompt  # Start with the requested analysis or revision prompt.
    for correction_round in range(2):  # Permit one schema or range correction within the same session.
        proposal = session.call_json(current_prompt, f"{purpose}_{correction_round + 1}")  # Obtain one live structured proposal.
        try:  # Validate it against deterministic model limits.
            configured, initial = validate_proposal(session.case_id, proposal)  # Convert only a valid proposal.
            return proposal, configured, initial  # Return immediately after validation passes.
        except (TypeError, ValueError, KeyError) as error:  # Preserve all proposal-contract failures.
            if correction_round == 1:  # Stop after the single allowed correction.
                raise RuntimeError(f"DeepSeek proposal remained invalid: {error}")  # Do not fabricate or clamp a configuration.
            current_prompt = (  # Send the exact deterministic validation failure back to the same session.
                f"你刚才的JSON未通过确定性合同：{type(error).__name__}: {error}。"
                f"请继续只处理案例{session.case_id}，按照原字段完整重发一个修正JSON。"
            )  # Complete the correction request.
    raise AssertionError("unreachable proposal correction state")  # Satisfy static control-flow analysis.


def run_case(case_id: str, entity_root: Path, output_root: Path, particles: int, iterations: int, seed: int) -> dict[str, Any]:  # Execute one complete isolated live session.
    output_root.mkdir(parents=True, exist_ok=True)  # Create the clean case output directory.
    if any(output_root.iterdir()):  # Require the caller to provide an empty from-scratch directory.
        raise RuntimeError(f"output directory is not empty: {output_root}")  # Prevent accidental trace or result reuse.
    session = DeepSeekSession(case_id, output_root)  # Create a new empty message history for this one case.
    session._write_trace(status="started")  # Persist the empty-history invariant before the first API call.
    first_proposal, preview_case, preview_position = proposal_with_correction(session, proposal_prompt(case_id), "initial_proposal")  # Obtain the first live configuration.
    brep_path = stage5.entity_path(preview_case, entity_root)  # Resolve or create the fixed exact entity for this case.
    preview_root = output_root / "preview"  # Allocate an isolated one-candidate probe directory.
    preview_evaluation = stage5.evaluate(preview_case, preview_position, brep_path, preview_root, keep=True)  # Perform one actual Gmsh remesh and audit.
    preview_payload = asdict(preview_evaluation)  # Serialize the actual probe evidence.
    revised_proposal, configured_case, initial_position = proposal_with_correction(  # Ask the same session to revise using real evidence.
        session,  # Continue the exact same local history.
        revision_prompt(case_id, preview_payload, first_proposal),  # Supply only this case's probe.
        "evidence_revision",  # Record call purpose in the trace.
    )  # Complete the revised proposal transaction.
    original_seeded_positions = stage5.seeded_positions  # Preserve the deterministic base seeding function.
    def seeded_positions_with_deepseek(case: stage5.Case, particle_count: int, random: Any) -> list[list[float]]:  # Inject the validated DeepSeek seed as particle zero.
        positions = original_seeded_positions(case, particle_count, random)  # Generate all standard physical and random seeds.
        positions[0] = list(initial_position)  # Replace only the first seed with the live case proposal.
        return positions  # Return the complete swarm without changing the PSO update law.
    stage5.seeded_positions = seeded_positions_with_deepseek  # Apply the case-local seed adapter for this process only.
    try:  # Ensure the imported module is restored even when optimization fails.
        pso_root = output_root / "pso"  # Allocate the full optimizer evidence directory.
        receipt = stage5.run_pso(configured_case, brep_path, pso_root, particles, iterations, seed)  # Execute all actual Gmsh candidates.
        stage5.write_svg(configured_case, pso_root / "best" / "candidate.msh", pso_root / "best" / "hotspot_mesh.svg")  # Render actual connectivity evidence.
    finally:  # Restore the deterministic module state.
        stage5.seeded_positions = original_seeded_positions  # Prevent accidental state leakage in local reuse.
    final_review = session.call_json(final_review_prompt(case_id, receipt["best"], receipt["history"]), "final_evidence_review")  # Obtain the final engineering interpretation in the same session.
    session._write_trace(status="completed")  # Seal the completed local conversation trace.
    session_trace = json.loads((output_root / "deepseek_session_trace.json").read_text(encoding="utf-8"))  # Read the final sealed trace.
    final_receipt = {  # Build the complete live case receipt.
        "schema_version": "entity-first-live-deepseek-case/1.0",  # Version the end-to-end case contract.
        "status": "completed",  # Mark completion only after PSO and final review.
        "case_id": case_id,  # Record the only case handled in this job.
        "session_id": session.session_id,  # Preserve the unique session identity.
        "started_from_empty_history": session.started_from_empty_history,  # Prove no shared V4 conversation was loaded.
        "deepseek_model": session.model,  # Record the live model requested.
        "deepseek_calls": len(session.calls),  # Record all proposal, correction, revision, and review calls.
        "first_call_prior_message_count": session.calls[0]["prior_message_count"],  # Prove the first request started without prior case history.
        "session_messages_sha256": session_trace["messages_sha256"],  # Seal the final local history.
        "initial_proposal": first_proposal,  # Preserve the first independent analysis.
        "preview_evaluation": preview_payload,  # Preserve one real pre-PSO remesh observation.
        "revised_proposal": revised_proposal,  # Preserve the evidence-informed final configuration.
        "configured_case": asdict(configured_case),  # Record exact bounds, ratios, and budget consumed by PSO.
        "deepseek_initial_position": list(initial_position),  # Record the live seed actually injected into particle zero.
        "pso_particles": particles,  # Record swarm population.
        "pso_iterations": iterations,  # Record optimizer depth.
        "pso_actual_unique_gmsh_evaluations": receipt["actual_unique_gmsh_evaluations"],  # Record actual remesh cost.
        "best": receipt["best"],  # Preserve deterministic topology, quality, density, and seam evidence.
        "best_score_history": [row["global_best_score"] for row in receipt["history"]],  # Preserve convergence evidence.
        "final_deepseek_review": final_review,  # Preserve the same session's final interpretation.
    }  # Complete the live case receipt.
    (output_root / "live_case_receipt.json").write_text(json.dumps(final_receipt, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist the final machine-readable case result.
    return final_receipt  # Return the completed case evidence.


def main() -> int:  # Execute one isolated live case from the command line.
    parser = argparse.ArgumentParser(description="Run one clean DeepSeek session with deterministic hotspot PSO")  # Describe the isolation contract.
    parser.add_argument("--case", choices=sorted(CASE_FACTS), required=True)  # Require exactly one model family.
    parser.add_argument("--entity-root", required=True)  # Require exact BREP evidence from the current workflow run.
    parser.add_argument("--output-dir", required=True)  # Require one clean case-isolated evidence directory.
    parser.add_argument("--particles", type=int, default=14)  # Default to the proven full swarm size.
    parser.add_argument("--iterations", type=int, default=12)  # Default to the proven full optimizer depth.
    parser.add_argument("--seed", type=int, default=20260801)  # Default to the reproducible deterministic PSO random stream.
    args = parser.parse_args()  # Parse command-line arguments once.
    if args.particles < 4 or args.iterations < 2:  # Preserve minimum meaningful PSO depth.
        parser.error("particles must be >=4 and iterations must be >=2")  # Reject a disguised shallow test.
    output_root = Path(args.output_dir).resolve()  # Normalize the case evidence directory.
    output_root.mkdir(parents=True, exist_ok=True)  # Create it before enforcing emptiness.
    receipt = run_case(args.case, Path(args.entity_root).resolve(), output_root, args.particles, args.iterations, args.seed)  # Execute the complete live session.
    print(json.dumps(receipt, ensure_ascii=False, indent=2))  # Echo the final receipt to GitHub Actions logs.
    return 0  # Return success only after live proposal, real remesh, full PSO, and final review pass.


if __name__ == "__main__":  # Run only when invoked directly.
    raise SystemExit(main())  # Propagate the process status to GitHub Actions.