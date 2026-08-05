const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { file: null, jobId: null, job: null, poll: null, relationFilter: "all" };

function esc(value = "") {
  return String(value).replace(/[&<>"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[char]));
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 3600);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function chooseFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) return toast("Choose a PDF paper.");
  state.file = file;
  $("#file-label").textContent = file.name;
  $("#run-button").disabled = false;
}

function stageIndex(stage) {
  return {queued: -1, parsing: 0, ner: 1, adjudicating: 2, resolving: 3, relationships: 4, canonicalizing: 5, matching: 6, neo4j: 7, complete: 8}[stage] ?? -1;
}

function stageOutput(output = {}) {
  return Object.entries(output).map(([key, value]) => {
    const rendered = Array.isArray(value) ? value.join(", ") : (typeof value === "object" && value !== null ? JSON.stringify(value) : value);
    return `${key.replaceAll("_", " ")}: ${rendered}`;
  }).join("\n");
}

function updateRun(job) {
  $("#run-panel").hidden = false;
  $("#run-message").textContent = job.message || "Working...";
  $("#run-percent").textContent = `${job.progress || 0}%`;
  $("#progress-bar").style.width = `${job.progress || 0}%`;
  const current = stageIndex(job.stage);
  const reports = new Map((job.stages || []).map(report => [report.stage, report]));
  $$("#stage-list li").forEach((li, index) => {
    const report = reports.get(li.dataset.stage);
    const done = report?.status === "complete";
    const active = report?.status === "running" && index === current && job.status === "running";
    li.classList.toggle("done", done);
    li.classList.toggle("active", active);
    const stageState = $(".stage-state", li);
    if (stageState) stageState.textContent = done ? "Completed" : active ? "Running" : (!report && job.status === "complete" ? "Skipped" : "Waiting");
    const output = $(".stage-output", li);
    if (output) {
      output.hidden = !report?.output;
      output.textContent = report?.output ? stageOutput(report.output) : "";
    }
  });
  const error = $("#run-error");
  error.hidden = job.status !== "failed";
  error.textContent = job.error || "";
}

async function begin(path, options = {}) {
  $("#results").hidden = true;
  $("#run-button").disabled = true;
  $$('[data-sample-button]').forEach(button => { button.disabled = true; });
  try {
    const job = await api(path, options);
    state.jobId = job.job_id;
    updateRun(job);
    pollJob();
  } catch (error) {
    toast(error.message);
    $("#run-button").disabled = !state.file;
    $$('[data-sample-button]').forEach(button => { button.disabled = false; });
  }
}

async function pollJob() {
  clearTimeout(state.poll);
  try {
    const job = await api(`/api/jobs/${state.jobId}`);
    state.job = job;
    updateRun(job);
    if (job.status === "complete") {
      renderResults(job);
      $("#run-button").disabled = !state.file;
      $$('[data-sample-button]').forEach(button => { button.disabled = false; });
      loadHistory();
      return;
    }
    if (job.status === "failed") {
      $("#run-button").disabled = !state.file;
      $$('[data-sample-button]').forEach(button => { button.disabled = false; });
      return;
    }
    state.poll = setTimeout(pollJob, 900);
  } catch (error) {
    toast(error.message);
    state.poll = setTimeout(pollJob, 1800);
  }
}

function metric(value, label, blue = false) {
  return `<div class="metric${blue ? " blue" : ""}"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`;
}

function renderResults(job) {
  const counts = job.summary.counts;
  $("#results").hidden = false;
  $("#paper-title").textContent = job.summary.paper.title || job.summary.source_filename;
  const llm = job.summary.llm;
  const runtime = job.summary.timing?.total_seconds;
  $("#run-subtitle").textContent = `${job.summary.parser_summary.sentences} narrative sentences | ${llm.used ? `local ${llm.model} check completed` : "deterministic node checks used"}${runtime ? ` | ${runtime.toFixed(1)} seconds` : ""}`;
  const graph = job.neo4j || job.summary.neo4j || {};
  const graphResult = $("#graph-result");
  if (graphResult) graphResult.textContent = graph.status === "upserted"
    ? `NEO4J UPDATED | ${graph.entities} paper nodes | ${graph.mentions} embedded mentions | ${graph.relationships} evidence-backed relationships | run ${graph.run_id}`
    : "Neo4j write receipt unavailable for this earlier run.";
  $("#bundle-download").href = `/api/jobs/${job.job_id}/bundle.zip`;
  const completeness = job.validation?.semantic_completeness || job.summary.semantic_completeness || {};
  const semanticResult = $("#semantic-result");
  if (semanticResult) {
    semanticResult.textContent = completeness.status
      ? `SEMANTIC COVERAGE ${completeness.status} | ${completeness.checks_passed} / ${completeness.checks_applicable} applicable checks | ${Math.round((completeness.coverage_score || 0) * 100)}%`
      : "Semantic coverage was not measured for this earlier run.";
    semanticResult.classList.toggle("warning", completeness.status === "INCOMPLETE");
  }
  $("#metrics").innerHTML = [
    ...(runtime ? [metric(runtime.toFixed(1), "pipeline seconds", true)] : []),
    metric(counts.accepted_mentions, "grounded mentions", true),
    metric(counts.canonical_entities, "canonical entities"),
    metric(counts.reference_mentions_resolved || 0, "resolved vague mentions", true),
    metric(counts.reference_review_pending || 0, "references to review"),
    metric(counts.node_review_candidates, "node candidates to review"),
    metric(`${counts.semantic_checks_passed || 0}/${counts.semantic_checks_applicable || 0}`, "semantic checks", true),
    metric(counts.similar_node_candidates, "possible duplicate pairs", true),
    metric(counts.relationships || 0, "relationships extracted"),
  ].join("");
  const documentId = job.summary.paper.id;
  state.graphQuery = `MATCH p=(source)-[relationship]-(target)\nWHERE relationship.documentId = '${documentId}'\nRETURN p\nLIMIT 150`;
  $("#graph-query").textContent = state.graphQuery;
  const relationshipCandidates = job.relationship_candidates || [];
  $("#relationship-badge").textContent = relationshipCandidates.length;
  $("#similar-badge").textContent = counts.similar_node_candidates;
  $("#review-badge").textContent = counts.node_review_candidates;
  $("#reference-badge").textContent = counts.reference_review_pending || 0;
  renderNodes(job.entities, job.mentions);
  renderRelationships(relationshipCandidates);
  renderQuestions(job.question_answerability || {questions: []});
  renderSimilar(job.similar_nodes);
  renderReferenceReview(job.reference_resolutions || []);
  renderReview(job.node_review_candidates);
  renderDownloads(job.downloads);
  refreshGraphHealth();
  $("#results").scrollIntoView({behavior: "smooth", block: "start"});
}

function renderQuestions(result) {
  const questions = result.questions || [];
  const answerable = questions.filter(row => row.status === "answerable").length;
  $("#question-badge").textContent = answerable;
  $("#question-coverage").textContent = `${answerable} / ${questions.length} answerable`;
  $("#question-list").innerHTML = questions.length ? questions.map(row => `
    <article class="review-card question-card">
      <div><span class="tag">${esc(row.status.replaceAll("_", " "))}</span><h4>${esc(row.question)}</h4>
      ${row.answers.length ? `<ul>${row.answers.map(answer => `<li>${esc(answer)}</li>`).join("")}</ul>` : `<p>No evidence-backed graph fact currently answers this question.</p>`}</div>
      <div class="question-evidence">${row.evidence.slice(0, 3).map(item => `<div><blockquote>"${esc(item.quote)}"</blockquote><span class="tag">page ${esc(item.pages.join(", ") || "unknown")}</span></div>`).join("")}</div>
    </article>`).join("") : `<div class="empty">This earlier run does not include a question-answerability evaluation.</div>`;
}

function gateLabel(value) {
  return ({
    endpoints_resolved: "unresolved endpoint",
    evidence_exact: "evidence mismatch",
    predicate_permitted: "predicate not permitted",
    domain_range: "domain/range mismatch",
    not_negated: "negated statement",
    not_modal: "modal language",
    not_hypothetical: "hypothetical statement",
    attribution_known: "unknown attribution",
    observation_structure: "missing observation structure",
  })[value] || value.replaceAll("_", " ");
}

function renderRelationships(rows) {
  state.relationshipCandidates = rows;
  const counts = {all: rows.length, accepted: 0, review: 0, rejected: 0};
  rows.forEach(row => { if (counts[row.status] !== undefined) counts[row.status] += 1; });
  $("#relation-all-count").textContent = counts.all;
  $("#relation-accepted-count").textContent = counts.accepted;
  $("#relation-review-count").textContent = counts.review;
  $("#relation-rejected-count").textContent = counts.rejected;
  const visible = rows.filter(row => state.relationFilter === "all" || row.status === state.relationFilter);
  $("#relationship-list").innerHTML = visible.length ? visible.map(row => {
    const predicate = row.llm_decision?.predicate || row.allowed_predicates?.[0] || "Pending";
    const failed = Object.entries(row.gates || {}).filter(([, pass]) => !pass).map(([gate]) => gateLabel(gate));
    return `<article class="relationship-card" data-status="${esc(row.status || "pending")}">
      <div class="relationship-triple">
        <div><small>${esc(row.subject_label)}</small><strong>${esc(row.subject_text)}</strong></div>
        <div class="predicate"><strong>${esc(predicate)}</strong><small>${esc(row.trigger_text ? `trigger: ${row.trigger_text}` : (row.channel || "").replaceAll("_", " "))}</small></div>
        <div><small>${esc(row.object_label)}</small><strong>${esc(row.object_text)}</strong></div>
      </div>
      <div class="relationship-meta"><span class="relation-status">${esc(row.status || "pending")}</span><span>${esc((row.attribution || "unknown").replaceAll("_", " "))}</span><span>${esc(failed.length ? `Gate: ${failed.join(", ")}` : "All gates passed")}</span></div>
      <details><summary>Evidence and decision</summary><blockquote>${esc(row.evidence_quote || "")}</blockquote><p>${esc(row.decision_reason || row.llm_decision?.reason || "Decision pending.")}</p></details>
    </article>`;
  }).join("") : `<div class="empty">${rows.length ? "No relationships match this filter." : "No relationship candidates were created for this run."}</div>`;
}

function renderNodes(entities, mentions) {
  const mentionMap = new Map(mentions.map(row => [row.mention_id, row]));
  state.nodeEntities = entities;
  state.mentionMap = mentionMap;
  const draw = () => {
    const query = normalized($("#node-search").value);
    const rows = entities.filter(row => normalized(`${row.canonical_name} ${row.label} ${row.aliases.join(" ")}`).includes(query));
    $("#node-count-label").textContent = `${rows.length} of ${entities.length}`;
    const root = $("#node-list");
    root.innerHTML = rows.length ? rows.map((row, index) => `
      <button class="node-row" type="button" data-entity="${esc(row.entity_id)}">
        <span class="node-dot">${esc(row.label.slice(0,2).toUpperCase())}</span>
        <span class="node-name"><strong>${esc(row.canonical_name)}</strong><small>${esc(row.label)} · ${row.aliases.length} alias${row.aliases.length === 1 ? "" : "es"}</small></span>
        <span class="node-count">×${row.mention_count}</span>
      </button>`).join("") : `<div class="empty">No nodes match this filter.</div>`;
    $$(".node-row", root).forEach(button => button.addEventListener("click", () => {
      $$(".node-row", root).forEach(row => row.classList.remove("active"));
      button.classList.add("active");
      const entity = entities.find(row => row.entity_id === button.dataset.entity);
      renderEvidence(entity, mentionMap);
    }));
    if (rows.length) $(".node-row", root).click();
  };
  $("#node-search").oninput = draw;
  draw();
}

function highlightedQuote(text, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) return esc(text);
  return `${esc(text.slice(0,start))}<mark>${esc(text.slice(start,end))}</mark>${esc(text.slice(end))}`;
}

function renderEvidence(entity, mentionMap) {
  const mention = entity.mention_ids.map(id => mentionMap.get(id)).find(row => row?.source?.kind === "sentence_span") || mentionMap.get(entity.mention_ids[0]);
  if (!mention) return;
  const source = mention.source || {};
  const context = source.context_sentences || [{role:"target", text:source.evidence_quote || mention.surface_text, sentence_id:source.sentence_id || "metadata"}];
  const resolvedReferences = entity.mention_ids.map(id => mentionMap.get(id)).filter(row => row?.reference_resolution);
  const resolvedReferenceHtml = resolvedReferences.length ? `
    <div class="context-label">Resolved surface mentions</div>
    ${resolvedReferences.map(row => {
      const resolution = row.reference_resolution;
      return `<div class="reference-evidence">
        <div class="evidence-meta"><span class="tag">${esc(resolution.status.replaceAll("_", " "))}</span><span class="tag">${Math.round((resolution.confidence || 0) * 100)}% confidence</span></div>
        <p><strong>“${esc(row.surface_text)}” → ${esc(resolution.target_canonical_name)}</strong></p>
        <blockquote>${esc(row.source.evidence_quote)}</blockquote>
        <small>Antecedent evidence</small><blockquote>${esc(resolution.antecedent_evidence_quote)}</blockquote>
      </div>`;
    }).join("")}` : "";
  $("#evidence-panel").innerHTML = `
    <p class="eyebrow">Traceability / ${esc(entity.label)}</p>
    <h3>${esc(entity.canonical_name)}</h3>
    <p class="muted">${esc(entity.aliases.join(" · "))}</p>
    <div class="evidence-meta"><span class="tag">${esc(mention.extraction_method)}</span><span class="tag">${esc(mention.validation?.judge || "traceable rule")}</span>${source.pages?.length ? `<span class="tag">page${source.pages.length > 1 ? "s" : ""} ${esc(source.pages.join(", "))}</span>` : ""}</div>
    <div class="context-label">Exact 2–3 sentence evidence window</div>
    ${context.map(item => `<div class="quote${item.role === "target" ? " target" : ""}"><span class="context-role">${esc(item.role)} sentence</span><br>${item.role === "target" ? highlightedQuote(item.text, source.start_char, source.end_char) : esc(item.text)}</div>`).join("")}
    <div class="trace-grid">
      <div><small>Source span</small><code>${esc(source.start_char ?? "—")} → ${esc(source.end_char ?? "—")}</code></div>
      <div><small>Section</small><code>${esc(source.section_title || source.field || "document metadata")}</code></div>
      <div><small>Sentence ID</small><code>${esc(source.sentence_id || "metadata")}</code></div>
      <div><small>Mention ID</small><code>${esc(mention.mention_id)}</code></div>
    </div>${resolvedReferenceHtml}`;
}

function normalized(value = "") { return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); }

function renderSimilar(pairs) {
  const root = $("#similar-list");
  if (!pairs.length) {
    root.innerHTML = `<div class="empty">No possible duplicates crossed the review threshold in this run.</div>`;
    return;
  }
  root.innerHTML = pairs.map(pair => `
    <article class="similar-card" data-resolution="${esc(pair.resolution_id)}">
      <div class="similar-head"><span>${esc(pair.label)}</span><span class="score">${Math.round(pair.score*100)}% similarity</span></div>
      <div class="pair">
        <div class="pair-node"><h4>${esc(pair.left.canonical_name)}</h4><p>${pair.left.mention_count} mention${pair.left.mention_count === 1 ? "" : "s"} · ${esc(pair.left.aliases.join(" · "))}</p></div>
        <div class="pair-vs">VS</div>
        <div class="pair-node"><h4>${esc(pair.right.canonical_name)}</h4><p>${pair.right.mention_count} mention${pair.right.mention_count === 1 ? "" : "s"} · ${esc(pair.right.aliases.join(" · "))}</p></div>
      </div>
      <div class="similar-foot">
        <span class="reasons">${esc(pair.reasons.join(" · "))}</span>
        ${pair.status === "pending" ? `
          <select aria-label="Canonical name"><option>${esc(pair.recommended_canonical_name)}</option>${[pair.left.canonical_name,pair.right.canonical_name].filter(name => name !== pair.recommended_canonical_name).map(name => `<option>${esc(name)}</option>`).join("")}</select>
          <button class="button quiet distinct-button" type="button">Keep separate</button>
          <button class="button blue same-button" type="button">Same node</button>` : `<span class="decision">Reviewed: ${esc(pair.status)}</span>`}
      </div>
    </article>`).join("");
  $$(".similar-card", root).forEach(card => {
    const same = $(".same-button", card);
    const distinct = $(".distinct-button", card);
    if (same) same.addEventListener("click", () => resolvePair(card, "same"));
    if (distinct) distinct.addEventListener("click", () => resolvePair(card, "distinct"));
  });
}

function renderReferenceReview(rows) {
  const root = $("#reference-list");
  if (!root) return;
  if (!rows.length) {
    root.innerHTML = `<div class="empty">No vague references need human resolution in this run.</div>`;
    return;
  }
  root.innerHTML = rows.map(row => {
    const candidates = row.candidate_targets || [];
    const selected = row.selected_target_mention_id || row.recommended_target_mention_id || candidates[0]?.target_mention_id || "";
    return `<article class="review-card reference-card" data-reference="${esc(row.resolution_id)}">
      <div><span class="tag">${esc(row.status)}</span><h4>“${esc(row.surface_text)}”</h4><p>${esc(row.reason || "Choose the supported antecedent.")}</p></div>
      <div><small>Exact surface-mention sentence</small><blockquote>“${esc(row.source.evidence_quote)}”</blockquote></div>
      <div class="reference-options">
        ${row.status === "pending" ? `<label><span>Resolved node</span><select>${candidates.map(candidate => `<option value="${esc(candidate.target_mention_id)}"${candidate.target_mention_id === selected ? " selected" : ""}>${esc(candidate.canonical_name)} · ${esc(candidate.label)} · ${candidate.sentence_distance === 0 ? "same sentence" : `${candidate.sentence_distance} sentence${candidate.sentence_distance === 1 ? "" : "s"} back`}</option>`).join("")}</select></label>` : `<strong>${esc(candidates.find(candidate => candidate.target_mention_id === row.selected_target_mention_id)?.canonical_name || "Ignored")}</strong>`}
        <div class="antecedent-preview">${candidates.map(candidate => `<details${candidate.target_mention_id === selected ? " open" : ""}><summary>${esc(candidate.canonical_name)}</summary><blockquote>${esc(candidate.antecedent_evidence_quote)}</blockquote></details>`).join("")}</div>
      </div>
      ${row.status === "pending" ? `<div class="review-actions"><button class="button quiet reference-ignore" type="button">Ignore reference</button><button class="button blue reference-resolve" type="button">Resolve to selected node</button></div>` : ""}
    </article>`;
  }).join("");
  $$(".reference-card", root).forEach(card => {
    $(".reference-resolve", card)?.addEventListener("click", () => resolveReference(card, "resolve"));
    $(".reference-ignore", card)?.addEventListener("click", () => resolveReference(card, "ignore"));
  });
}

async function resolveReference(card, decision) {
  const buttons = $$("button", card);
  buttons.forEach(button => button.disabled = true);
  try {
    await api(`/api/jobs/${state.jobId}/reference-resolution`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        resolution_id: card.dataset.reference,
        decision,
        target_mention_id: $("select", card)?.value || "",
      }),
    });
    toast(decision === "resolve" ? "The surface mention now traces to the selected node." : "The unresolved reference was ignored.");
    state.job = await api(`/api/jobs/${state.jobId}`);
    renderResults(state.job);
    activateTab("references");
  } catch (error) {
    toast(error.message);
    buttons.forEach(button => button.disabled = false);
  }
}

async function resolvePair(card, decision) {
  const buttons = $$("button", card);
  buttons.forEach(button => button.disabled = true);
  try {
    const chosen = $("select", card)?.value || "";
    await api(`/api/jobs/${state.jobId}/resolution`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({resolution_id: card.dataset.resolution, decision, chosen_canonical_name: chosen}),
    });
    toast(decision === "same" ? `Merged under “${chosen}” and added aliases to the persistent lexicon.` : "Recorded as distinct nodes.");
    state.job = await api(`/api/jobs/${state.jobId}`);
    renderResults(state.job);
    activateTab("similar");
  } catch (error) {
    toast(error.message);
    buttons.forEach(button => button.disabled = false);
  }
}

function renderReview(rows) {
  const root = $("#review-list");
  const labels = state.job?.ontology_labels || [];
  root.innerHTML = rows.length ? rows.map(row => `
    <article class="review-card node-review-card" data-candidate="${esc(row.candidate_id)}">
      <div><span class="tag">${esc(row.label || "untyped")}</span><h4>${esc(row.canonical_name || row.surface_text)}</h4><p>${esc(row.extraction_method)}</p><small>Sentence ID: ${esc(row.source?.sentence_id || "MISSING")}</small></div>
      <div><blockquote>"${esc(row.source?.evidence_quote || row.surface_text)}"</blockquote><span class="tag">page ${esc(row.source?.pages?.join(", ") || "unknown")}</span></div>
      <div class="node-review-fields">
        <label><span>Ontology type</span><select class="node-review-label">${labels.map(label => `<option value="${esc(label)}"${label === row.label ? " selected" : ""}>${esc(label)}</option>`).join("")}</select></label>
        <label><span>Canonical name</span><input class="node-review-name" value="${esc(row.canonical_name || row.surface_text)}"></label>
      </div>
      <div class="review-actions"><button class="button quiet node-reject" type="button">Reject</button><button class="button blue node-accept" type="button">Accept node</button></div>
    </article>`).join("") : `<div class="empty">No additional node phrases need human approval.</div>`;
  $$(".node-review-card", root).forEach(card => {
    $(".node-accept", card)?.addEventListener("click", () => submitNodeReview(card, "accept"));
    $(".node-reject", card)?.addEventListener("click", () => submitNodeReview(card, "reject"));
  });
}

async function submitNodeReview(card, decision) {
  const buttons = $$("button", card);
  buttons.forEach(button => button.disabled = true);
  try {
    const result = await api(`/api/jobs/${state.jobId}/node-review`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        candidate_id: card.dataset.candidate,
        decision,
        ontology_label: $(".node-review-label", card)?.value || "",
        canonical_name: $(".node-review-name", card)?.value || "",
      }),
    });
    toast(result.decision === "accept" ? "Node accepted, added to the graph, and recorded in the persistent lexicon." : "Node rejected and excluded from the graph.");
    state.job = await api(`/api/jobs/${state.jobId}`);
    renderResults(state.job);
    activateTab("review");
  } catch (error) {
    toast(error.message);
    buttons.forEach(button => button.disabled = false);
  }
}

function renderDownloads(files) {
  const descriptions = {
    "parsed.json":"Immutable narrative-text parse", "grammatical_analysis.jsonl":"Complete token, POS, morphology, and dependency analysis",
    "grammatical_triples.jsonl":"Predicate-first subject-predicate-object triples", "scibert_typings.jsonl":"Pretrained SciBERT ontology rankings",
    "grammatical_stage.json":"Parser and SciBERT stage summary", "node_candidates.jsonl":"All high-recall node candidates",
    "node_normalization_candidates.jsonl":"Deterministic cleanup plus same-type prior-node comparison options",
    "candidate_judgments.jsonl":"Every accept, review, and reject decision", "mentions.jsonl":"Accepted mentions + exact evidence",
    "node_review_candidates.jsonl":"Compatibility copy of the review queue", "review_queue.jsonl":"Candidates awaiting human approval",
    "node_rejections.jsonl":"Rejected candidates with reasons", "canonical_entities.jsonl":"Pre-review canonical nodes",
    "reference_resolution_review.jsonl":"Ambiguous vague mentions with finite antecedent choices",
    "reference_resolution_ignored.jsonl":"Unresolved references excluded from the graph",
    "canonical_entities_merged.jsonl":"Human-merged canonical nodes", "similar_nodes_review.jsonl":"Similarity features + decisions",
    "lexicon_snapshot.json":"Lexicon used for this paper", "llm_node_judge_calls.jsonl":"Node-candidate local-model audit trail",
    "llm_reference_judge_calls.jsonl":"Contextual SAME/DIFFERENT reference-pair audit trail",
    "relationship_candidates.jsonl":"All context-window LLM proposals, endpoint selections, judgments, and gates",
    "relationship_unknown_nodes.jsonl":"Relationship endpoints absent from the accepted node catalog",
    "assertions.jsonl":"Accepted evidence-backed relationship assertions",
    "relationship_rejections.jsonl":"Rejected candidates and failed gates",
    "llm_relationship_judge_calls.jsonl":"Context-window relationship extraction and judgment audit trail",
    "canonical_relationships.jsonl":"Canonical triples aggregated from accepted assertions",
    "question_answerability.json":"Graph-only paper questions, answers, and exact supporting sentences",
    "stage_outputs.json":"Status and output summary for every pipeline stage",
    "semantic_completeness.json":"Recall-oriented checks for introduced systems, agents, models, connectivity, and traceability",
    "neo4j_upsert.json":"Verified Neo4j write receipt",
    "validation_report.json":"Traceability and mode checks", "manifest.json":"Run summary and file index", "source.pdf":"Original paper",
  };
  $("#download-grid").innerHTML = files.map(file => `
    <a class="download-card" href="/api/jobs/${state.jobId}/download/${encodeURIComponent(file)}">
      <strong>${esc(file)}</strong><small>${esc(descriptions[file] || "Run artifact")}</small><span>Download ↓</span>
    </a>`).join("");
}

function activateTab(name) {
  $$(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.tab === name));
  $$(".tab-panel").forEach(panel => panel.classList.toggle("active", panel.id === `panel-${name}`));
}

async function loadHistory() {
  try {
    const payload = await api("/api/jobs");
    const select = $("#history-select");
    select.innerHTML = `<option value="">Previous runs</option>` + payload.jobs.filter(row => row.status === "complete").map(row => {
      const source = (row.summary?.paper?.title || row.summary?.source_filename || row.job_id).replace(/\.pdf$/i, "");
      const paper = source.length > 52 ? `${source.slice(0, 49)}...` : source;
      const stamp = new Date(row.summary?.created_at || row.created_at);
      const when = Number.isNaN(stamp.getTime()) ? "time unavailable" : stamp.toLocaleString([], {month:"short", day:"numeric", hour:"numeric", minute:"2-digit"});
      const shortId = row.job_id.replace(/^run_/, "").slice(0, 15);
      return `<option value="${esc(row.job_id)}">${esc(`${paper} | ${when} | ${shortId}`)}</option>`;
    }).join("");
  } catch (_) {}
}

async function refreshGraphHealth() {
  if (!$("#graph-health")) return;
  try {
    const graph = await api("/api/graph/summary");
    const exact = graph.mentions === graph.grounded_mentions;
    const references = graph.resolved_mentions === graph.linked_antecedents;
    const ontologyOnly = graph.operational_instance_nodes === 0 && graph.duplicate_canonical_keys === 0;
    const paperOnly = ontologyOnly && graph.materialized_ontology_nodes === 0;
    $("#graph-health-title").textContent = `${graph.canonical_entities} paper nodes and ${graph.semantic_relationships} extracted relationships`;
    const confidence = graph.nodes_with_confidence === 0 ? "no confidence node properties" : `${graph.nodes_with_confidence} nodes still carry confidence`;
    $("#graph-health-detail").textContent = `${graph.materialized_ontology_nodes} materialized ontology nodes · ${graph.grounded_mentions}/${graph.mentions} mentions retain exact evidence · ${graph.linked_antecedents}/${graph.resolved_mentions} resolved references traced · ${confidence}${exact && references && paperOnly ? " · paper-only graph complete" : " · review needed"}`;
  } catch (_) {
    $("#graph-health-title").textContent = "Neo4j is unavailable";
    $("#graph-health-detail").textContent = "Start the local graph, then refresh this page.";
  }
}

async function init() {
  try {
    const status = await api("/api/status");
    $("#footer-status").textContent = `Paper nodes + predicates + Neo4j | ${status.lexicon_entries} lexicon entries | graph on :7477`;
    await refreshGraphHealth();
  } catch (_) {
    $("#footer-status").textContent = "Workbench service unavailable";
    if ($("#graph-health-title")) $("#graph-health-title").textContent = "Neo4j is unavailable";
    if ($("#graph-health-detail")) $("#graph-health-detail").textContent = "Start the local graph, then refresh this page.";
  }
  loadHistory();
}

$("#choose-button").addEventListener("click", () => $("#pdf-input").click());
$("#pdf-input").addEventListener("change", event => chooseFile(event.target.files[0]));
const dropzone = $("#dropzone");
["dragenter","dragover"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave","drop"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));
$("#run-button").addEventListener("click", () => {
  const form = new FormData(); form.append("paper", state.file); begin("/api/extract", {method:"POST", body:form});
});
$("#sample-button").addEventListener("click", () => begin("/api/extract/sample", {method:"POST"}));
$("#epanet-sample-button").addEventListener("click", () => begin("/api/extract/sample/epanet-agentic", {method:"POST"}));
$("#copy-graph-query").addEventListener("click", async () => {
  if (!state.graphQuery) return;
  await navigator.clipboard.writeText(state.graphQuery);
  toast("Neo4j query copied. Open Neo4j and press Ctrl+V, then Ctrl+Enter.");
});
$("#history-select").addEventListener("change", async event => {
  if (!event.target.value) return;
  state.jobId = event.target.value;
  state.job = await api(`/api/jobs/${state.jobId}`);
  updateRun(state.job); renderResults(state.job);
});
$$(".tab").forEach(tab => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
$$(".relation-filter").forEach(button => button.addEventListener("click", () => {
  state.relationFilter = button.dataset.relationFilter;
  $$(".relation-filter").forEach(item => item.classList.toggle("active", item === button));
  renderRelationships(state.relationshipCandidates || []);
}));
init();
