const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = {
  file: null,
  jobId: null,
  job: null,
  poll: null,
  relationFilter: "all",
  llmModel: "",
  modelsReady: false,
};

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
  $("#run-button").disabled = !state.modelsReady;
}

function modelOptionLabel(model) {
  const detail = [model.parameter_size, model.quantization].filter(Boolean).join(" · ");
  return detail ? `${model.name} — ${detail}` : model.name;
}

async function loadModels() {
  const select = $("#model-select");
  const help = $("#model-help");
  try {
    const payload = await api("/api/models");
    const models = payload.models || [];
    state.modelsReady = Boolean(payload.available && models.length);
    select.innerHTML = models.length
      ? models.map(model => `<option value="${esc(model.name)}">${esc(modelOptionLabel(model))}</option>`).join("")
      : `<option value="">No Ollama models found</option>`;
    state.llmModel = models.some(model => model.name === payload.default_model)
      ? payload.default_model
      : (models[0]?.name || "");
    select.value = state.llmModel;
    select.disabled = !state.modelsReady;
    help.textContent = state.modelsReady
      ? "Used for node and relationship decisions in this run."
      : (payload.error || "Start Ollama and install the configured Qwen model.");
    $("#run-button").disabled = !(state.file && state.modelsReady);
    $$('[data-sample-button]').forEach(button => { button.disabled = !state.modelsReady; });
  } catch (error) {
    state.modelsReady = false;
    select.innerHTML = `<option value="">Models unavailable</option>`;
    select.disabled = true;
    help.textContent = error.message;
    $("#run-button").disabled = true;
    $$('[data-sample-button]').forEach(button => { button.disabled = true; });
  }
}

function stageIndex(stage) {
  return {queued: -1, parsing: 0, ner: 1, adjudicating: 2, resolving: 3, relationships: 4, complete: 5}[stage] ?? -1;
}

function stageOutput(output = {}) {
  return Object.entries(output).map(([key, value]) => {
    const rendered = Array.isArray(value) ? value.join(", ") : (typeof value === "object" && value !== null ? JSON.stringify(value) : value);
    return `${key.replaceAll("_", " ")}: ${rendered}`;
  }).join("\n");
}

function number(value) {
  return Number(value || 0).toLocaleString();
}

function duration(seconds) {
  const total = Math.round(Number(seconds || 0));
  if (total < 60) return `${total} seconds`;
  if (total < 3600) return `${Math.floor(total / 60)} min ${total % 60} sec`;
  return `${Math.floor(total / 3600)} hr ${Math.round((total % 3600) / 60)} min`;
}

function plainLabel(value = "") {
  return String(value)
    .replaceAll("_", " ")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function stageSummary(stage, output = {}) {
  const summaries = {
    parsing: () => `${number(output.sentences)} sentences read across ${number(output.narrative_pages || output.pages)} pages`,
    ner: () => `${number(output.triples)} raw subject-verb-object candidates from ${number(output.sentences)} sentences`,
    adjudicating: () => `${number(output.candidates)} candidates scored; ${number(output.untyped)} returned no SciBERT type`,
    resolving: () => `${number(output.accepted_mentions)} supported items kept; ${number(output.needs_review)} need review`,
    relationships: () => `${number(output.accepted || output.accepted_assertions || output.assertions)} supported connections kept`,
  };
  return summaries[stage]?.() || "Step completed";
}

function runMessage(job) {
  if (job.status === "failed") return "The run stopped";
  if (job.status === "complete") return "Results are ready to review";
  return ({
    queued: "Waiting to start",
    parsing: "Reading the paper",
    ner: "Extracting grammatical candidates",
    adjudicating: "Scoring ontology types",
    resolving: "Reviewing node candidates",
    relationships: "Extracting connections",
  })[job.stage] || "Working on the paper";
}

function updateRun(job) {
  if (job.status === "complete" && !(job.stages || []).length) {
    $("#run-panel").hidden = true;
    return;
  }
  $("#run-panel").hidden = false;
  $("#run-message").textContent = runMessage(job);
  const runDetails = [];
  if (job.llm_model) runDetails.push(`Model: ${job.llm_model}`);
  if (job.processing_unit) runDetails.push(`Relationships: ${job.processing_unit}`);
  $("#run-model").textContent = runDetails.join(" · ");
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
    const summary = $(".stage-summary", li);
    if (summary) summary.textContent = done && report?.output ? stageSummary(report.stage, report.output) : "";
    const details = $(".stage-details", li);
    if (details) details.hidden = !report?.output;
    const output = $(".stage-output", li);
    if (output) output.textContent = report?.output ? stageOutput(report.output) : "";
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
    $("#run-button").disabled = !(state.file && state.modelsReady);
    $$('[data-sample-button]').forEach(button => { button.disabled = !state.modelsReady; });
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
      $("#run-button").disabled = !(state.file && state.modelsReady);
      $$('[data-sample-button]').forEach(button => { button.disabled = !state.modelsReady; });
      loadHistory();
      return;
    }
    if (job.status === "failed") {
      $("#run-button").disabled = !(state.file && state.modelsReady);
      $$('[data-sample-button]').forEach(button => { button.disabled = !state.modelsReady; });
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
  const counts = job.summary.counts || {};
  const relationshipCandidates = job.relationship_candidates || [];
  const questions = job.question_answerability?.questions || [];
  const answeredQuestions = questions.filter(row => row.status === "answerable").length;
  const connectionReviews = relationshipCandidates.filter(row => row.status === "needs_review").length;
  const reviewCount = (counts.node_review_candidates || 0) + connectionReviews;
  $("#results").hidden = false;
  const document = job.summary.document || {};
  $("#paper-title").textContent = document.title || document.filename || "Paper results";
  const runtime = job.summary.timing?.total_seconds;
  $("#run-subtitle").textContent = `${number(job.summary.parser_summary.sentences)} written sentences reviewed${runtime ? ` in ${duration(runtime)}` : ""}.`;
  renderGraphPublication(job);
  $("#bundle-download").href = `/api/jobs/${job.job_id}/bundle.zip`;
  const completeness = job.validation?.semantic_completeness || job.summary.semantic_completeness || {};
  const semanticResult = $("#semantic-result");
  if (semanticResult) {
    semanticResult.textContent = completeness.status
      ? `${completeness.checks_passed} of ${completeness.checks_applicable} coverage checks passed (${Math.round((completeness.coverage_score || 0) * 100)}%).`
      : "Coverage checks were not available for this earlier run.";
    semanticResult.classList.toggle("warning", completeness.status === "INCOMPLETE");
  }
  $("#metrics").innerHTML = [
    metric(counts.canonical_entities || 0, "items found", true),
    metric(counts.relationships || 0, "supported connections"),
    metric(reviewCount, "decisions needed", reviewCount > 0),
    metric(answeredQuestions || counts.answerable_questions || 0, "questions answered"),
  ].join("");
  $("#technical-metrics").innerHTML = [
    metric(counts.accepted_mentions || 0, "supported text references"),
    metric(counts.rejected_node_candidates || 0, "items excluded"),
    metric(`${counts.semantic_checks_passed || 0}/${counts.semantic_checks_applicable || 0}`, "coverage checks"),
    ...(runtime ? [metric(runtime.toFixed(1), "processing seconds")] : []),
  ].join("");
  const documentId = document.id || "";
  state.graphQuery = `MATCH p=(source)-[relationship]-(target)\nWHERE relationship.documentId = '${documentId}'\nRETURN p\nLIMIT 150`;
  $("#graph-query").textContent = state.graphQuery;
  $("#relationship-badge").textContent = relationshipCandidates.length;
  $("#review-badge").textContent = counts.node_review_candidates || 0;
  renderNodes(job.entities, job.mentions);
  renderRelationships(relationshipCandidates);
  renderQuestions(job.question_answerability || {questions: []});
  renderReview(job.node_review_candidates);
  renderDownloads(job.downloads);
  refreshGraphHealth();
  $("#results").scrollIntoView({behavior: "smooth", block: "start"});
}

function renderGraphPublication(job) {
  const publication = job.graph_publication || job.summary.graph_publication || {state: "not_added", add_count: 0};
  const graph = job.neo4j || job.summary.neo4j || {};
  const stateNode = $("#graph-publication-state");
  const stateLabel = $("#graph-publication-label");
  const detail = $("#graph-publication-detail");
  const addButton = $("#add-to-graph");
  const removeButton = $("#remove-from-graph");
  const graphLink = $("#open-run-graph");
  const labels = {not_added: "Not saved", added: "Saved", removed: "Removed"};
  stateNode.dataset.state = publication.state;
  stateLabel.textContent = labels[publication.state] || publication.state;
  addButton.hidden = publication.state !== "not_added";
  removeButton.hidden = publication.state !== "added";
  graphLink.hidden = publication.state !== "added";
  graphLink.href = publication.graph_url || "http://127.0.0.1:7477/";
  addButton.disabled = false;
  removeButton.disabled = false;

  if (publication.state === "added") {
    detail.textContent = "This paper's reviewed results are in the shared graph. It cannot be saved a second time.";
  } else if (publication.state === "removed") {
    detail.textContent = "This paper was removed from the shared graph. Because it was saved once already, it cannot be saved again.";
  } else {
    detail.textContent = "Review is complete. Save this paper once, or leave the shared graph unchanged.";
  }

  const graphResult = $("#graph-result");
  if (!graphResult) return;
  if (publication.state === "added") {
    graphResult.textContent = "Saved to the shared graph.";
  } else if (publication.state === "removed") {
    graphResult.textContent = "Removed from the shared graph.";
  } else {
    graphResult.textContent = "Ready to review. Not yet saved to the shared graph.";
  }
}

async function graphAction(action) {
  if (!state.jobId) return;
  const addButton = $("#add-to-graph");
  const removeButton = $("#remove-from-graph");
  addButton.disabled = true;
  removeButton.disabled = true;
  try {
    await api(`/api/jobs/${state.jobId}/graph/${action}`, {method: "POST"});
    state.job = await api(`/api/jobs/${state.jobId}`);
    renderGraphPublication(state.job);
    renderDownloads(state.job.downloads || []);
    await refreshGraphHealth();
    toast(action === "add" ? "Paper saved to the shared graph." : "Paper removed from the shared graph.");
  } catch (error) {
    toast(error.message);
    addButton.disabled = false;
    removeButton.disabled = false;
  }
}

function renderQuestions(result) {
  const questions = result.questions || [];
  const answerable = questions.filter(row => row.status === "answerable").length;
  $("#question-badge").textContent = answerable;
  $("#question-coverage").textContent = `${answerable} of ${questions.length} answered`;
  $("#question-list").innerHTML = questions.length ? questions.map(row => `
    <article class="review-card question-card">
      <div><span class="tag">${row.status === "answerable" ? "answered" : "not answered"}</span><h4>${esc(row.question)}</h4>
      ${row.answers.length ? `<ul>${row.answers.map(answer => `<li>${esc(answer)}</li>`).join("")}</ul>` : `<p>The reviewed results do not currently answer this question.</p>`}</div>
      <div class="question-evidence">${row.evidence.slice(0, 3).map(item => `<div><blockquote>"${esc(item.quote)}"</blockquote><span class="tag">page ${esc(item.pages.join(", ") || "unknown")}</span></div>`).join("")}</div>
    </article>`).join("") : `<div class="empty">No paper questions are available for this run.</div>`;
}

function gateLabel(value) {
  return ({
    endpoints_resolved: "one of the items could not be identified",
    evidence_exact: "the source sentence did not match",
    predicate_permitted: "the connection type was not allowed",
    domain_range: "the two item types do not support this connection",
    not_negated: "the paper says this did not happen",
    not_modal: "the paper presents this only as a possibility",
    not_hypothetical: "the paper presents this as hypothetical",
    attribution_known: "the source of the claim is unclear",
    observation_structure: "the observation is incomplete",
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
    const predicate = plainLabel(row.llm_decision?.predicate || row.allowed_predicates?.[0] || "Not decided");
    const failed = Object.entries(row.gates || {}).filter(([, pass]) => !pass).map(([gate]) => gateLabel(gate));
    const status = ({accepted: "supported", review: "needs review", rejected: "excluded"})[row.status] || "not decided";
    return `<article class="relationship-card" data-status="${esc(row.status || "pending")}">
      <div class="relationship-triple">
        <div><small>${esc(plainLabel(row.subject_label))}</small><strong>${esc(row.subject_text)}</strong></div>
        <div class="predicate"><small>connection</small><strong>${esc(predicate)}</strong></div>
        <div><small>${esc(plainLabel(row.object_label))}</small><strong>${esc(row.object_text)}</strong></div>
      </div>
      <div class="relationship-meta"><span class="relation-status">${esc(status)}</span>${failed.length ? `<span>${esc(failed.join("; "))}</span>` : ""}</div>
      <details><summary>Source and explanation</summary><blockquote>${esc(row.evidence_quote || "")}</blockquote><p>${esc(row.decision_reason || row.llm_decision?.reason || "No explanation is available.")}</p></details>
      <details class="technical-disclosure"><summary>Technical details</summary><p>Method: ${esc(plainLabel(row.channel || "unknown"))}<br>Trigger: ${esc(row.trigger_text || "not recorded")}<br>Attribution: ${esc(plainLabel(row.attribution || "unknown"))}</p></details>
    </article>`;
  }).join("") : `<div class="empty">${rows.length ? "No connections match this filter." : "No connections were found in this run."}</div>`;
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
        <span class="node-name"><strong>${esc(row.canonical_name)}</strong><small>${esc(plainLabel(row.label))}</small></span>
        <span class="node-count">${row.mention_count} mention${row.mention_count === 1 ? "" : "s"}</span>
      </button>`).join("") : `<div class="empty">No items match this filter.</div>`;
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
  $("#evidence-panel").innerHTML = `
    <p class="eyebrow">Source / ${esc(plainLabel(entity.label))}</p>
    <h3>${esc(entity.canonical_name)}</h3>
    ${entity.aliases.length ? `<p class="muted">Also written as ${esc(entity.aliases.join(", "))}</p>` : ""}
    <div class="evidence-meta">${source.pages?.length ? `<span class="tag">page${source.pages.length > 1 ? "s" : ""} ${esc(source.pages.join(", "))}</span>` : ""}</div>
    <div class="context-label">Supporting sentence and context</div>
    ${context.map(item => `<div class="quote${item.role === "target" ? " target" : ""}"><span class="context-role">${item.role === "target" ? "supporting sentence" : "context"}</span><br>${item.role === "target" ? highlightedQuote(item.text, source.start_char, source.end_char) : esc(item.text)}</div>`).join("")}
    <details class="technical-disclosure"><summary>Technical details</summary>
      <p>Method: ${esc(plainLabel(mention.extraction_method))}<br>Check: ${esc(plainLabel(mention.validation?.judge || "traceable rule"))}</p>
      <div class="trace-grid">
        <div><small>Character range</small><code>${esc(source.start_char ?? "—")} → ${esc(source.end_char ?? "—")}</code></div>
        <div><small>Section</small><code>${esc(source.section_title || source.field || "document metadata")}</code></div>
        <div><small>Sentence ID</small><code>${esc(source.sentence_id || "metadata")}</code></div>
        <div><small>Item ID</small><code>${esc(mention.mention_id)}</code></div>
      </div>
    </details>`;
}

function normalized(value = "") { return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); }

function renderReview(rows) {
  const root = $("#review-list");
  const labels = state.job?.ontology_labels || [];
  root.innerHTML = rows.length ? rows.map(row => `
    <article class="review-card node-review-card" data-candidate="${esc(row.candidate_id)}">
      <div><span class="tag">${esc(plainLabel(row.label || "type not chosen"))}</span><h4>${esc(row.canonical_name || row.surface_text)}</h4><details class="technical-disclosure"><summary>Technical details</summary><p>Method: ${esc(plainLabel(row.extraction_method))}<br>Sentence ID: ${esc(row.source?.sentence_id || "missing")}</p></details></div>
      <div><blockquote>"${esc(row.source?.evidence_quote || row.surface_text)}"</blockquote><span class="tag">page ${esc(row.source?.pages?.join(", ") || "unknown")}</span></div>
      <div class="node-review-fields">
        <label><span>Item type</span><select class="node-review-label">${labels.map(label => `<option value="${esc(label)}"${label === row.label ? " selected" : ""}>${esc(plainLabel(label))}</option>`).join("")}</select></label>
        <label><span>Name to use</span><input class="node-review-name" value="${esc(row.canonical_name || row.surface_text)}"></label>
      </div>
      <div class="review-actions"><button class="button quiet node-reject" type="button">Exclude</button><button class="button blue node-accept" type="button">Keep item</button></div>
    </article>`).join("") : `<div class="empty">No other items need review.</div>`;
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
    toast(result.decision === "accept" ? "The item was kept." : "The item was excluded.");
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
    "parsed.json":"Immutable narrative-text parse",
    "raw_svo.jsonl":"Raw main-clause SVO plus separate sentence context",
    "grammatical_triples.jsonl":"Raw subject-verb-object triples",
    "paragraph_batches.jsonl":"Paragraph batches with subjects, objects, and context separated",
    "scibert_typings.jsonl":"Advisory SciBERT ontology rankings, including NONE",
    "node_candidates.jsonl":"Raw SVO candidates and three-pass node decisions",
    "candidate_judgments.jsonl":"Every accepted, review, and disregard decision",
    "mentions.jsonl":"Accepted mentions with exact evidence",
    "review_queue.jsonl":"Nodes awaiting human approval",
    "node_rejections.jsonl":"Disregarded nodes with reasons",
    "canonical_entities_merged.jsonl":"Canonical accepted nodes",
    "ollama_node_calls.jsonl":"Three-pass Qwen node audit trail",
    "relationship_candidates.jsonl":"Generated, refined, validated paragraph relationships",
    "relationship_unknown_nodes.jsonl":"Inferred endpoints awaiting human approval",
    "assertions.jsonl":"Accepted evidence-backed relationship assertions",
    "relationship_rejections.jsonl":"Rejected relationships and failed gates",
    "ollama_relationship_calls.jsonl":"Qwen generation, refinement, and validation audit trail",
    "canonical_relationships.jsonl":"Canonical triples aggregated from accepted assertions",
    "question_answerability.json":"Graph questions and supporting sentences",
    "semantic_completeness.json":"Recall-oriented completeness checks",
    "generalization_diagnostics.json":"Graph coverage and quality diagnostics",
    "neo4j_upsert.json":"Verified Neo4j write receipt",
    "neo4j_removal.json":"Verified per-paper Neo4j removal receipt",
    "validation_report.json":"Deterministic provenance checks",
    "manifest.json":"Run summary and file index",
    "source.pdf":"Original paper",
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
    const visibleJobs = payload.jobs.filter(row => row.status !== "failed");
    select.innerHTML = `<option value="">Previous runs</option>` + visibleJobs.map(row => {
      const source = (row.summary?.paper?.title || row.summary?.source_filename || row.job_id).replace(/\.pdf$/i, "");
      const paper = source.length > 52 ? `${source.slice(0, 49)}...` : source;
      const stamp = new Date(row.summary?.created_at || row.created_at);
      const when = Number.isNaN(stamp.getTime()) ? "time unavailable" : stamp.toLocaleString([], {month:"short", day:"numeric", hour:"numeric", minute:"2-digit"});
      const shortId = row.job_id.replace(/^run_/, "").slice(0, 15);
      const status = row.status === "complete" ? "" : ` | ${row.status.toUpperCase()}`;
      return `<option value="${esc(row.job_id)}">${esc(`${paper} | ${when} | ${shortId}${status}`)}</option>`;
    }).join("");
    const active = payload.jobs.find(row => row.status === "running" || row.status === "queued");
    if (!state.jobId && active) {
      state.jobId = active.job_id;
      select.value = active.job_id;
      $("#results").hidden = true;
      $("#run-button").disabled = true;
      $$('[data-sample-button]').forEach(button => { button.disabled = true; });
      updateRun(active);
      pollJob();
    }
  } catch (_) {}
}

async function refreshGraphHealth() {
  if (!$("#graph-health")) return;
  try {
    const graph = await api("/api/graph/summary");
    const exact = graph.mentions === graph.grounded_mentions;
    const ontologyOnly = graph.operational_instance_nodes === 0 && graph.duplicate_canonical_keys === 0;
    const paperOnly = ontologyOnly && graph.materialized_ontology_nodes === 0;
    $("#graph-health-title").textContent = `${graph.canonical_entities} paper nodes and ${graph.semantic_relationships} extracted relationships`;
    const confidence = graph.nodes_with_confidence === 0 ? "no confidence node properties" : `${graph.nodes_with_confidence} nodes still carry confidence`;
    $("#graph-health-detail").textContent = `${graph.materialized_ontology_nodes} materialized ontology nodes · ${graph.grounded_mentions}/${graph.mentions} mentions retain exact evidence · ${confidence}${exact && paperOnly ? " · paper-only graph complete" : " · review needed"}`;
  } catch (_) {
    $("#graph-health-title").textContent = "Neo4j is unavailable";
    $("#graph-health-detail").textContent = "Start the local graph, then refresh this page.";
  }
}

function closeResetDialog() {
  const dialog = $("#reset-graph-dialog");
  if (dialog.open) dialog.close();
  $("#reset-confirmation").value = "";
  $("#confirm-graph-reset").disabled = true;
}

async function resetSharedGraph() {
  const confirmation = $("#reset-confirmation").value;
  if (confirmation !== "RESET") return;
  const confirmButton = $("#confirm-graph-reset");
  confirmButton.disabled = true;
  confirmButton.textContent = "Removing graph data...";
  try {
    const result = await api("/api/graph/reset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({confirmation}),
    });
    closeResetDialog();
    const receipt = result.receipt || {};
    toast(`Shared graph reset: ${number(receipt.nodes_removed)} nodes and ${number(receipt.relationships_removed)} relationships removed.`);
    await refreshGraphHealth();
    await loadHistory();
    if (state.jobId) {
      state.job = await api(`/api/jobs/${state.jobId}`);
      renderResults(state.job);
    }
  } catch (error) {
    toast(error.message);
    confirmButton.disabled = $("#reset-confirmation").value !== "RESET";
  } finally {
    confirmButton.textContent = "Remove all graph data";
  }
}

async function init() {
  await loadModels();
  try {
    const status = await api("/api/status");
    const modelsReady = Boolean(status.model_readiness && status.model_readiness.ready);
    const memory = `${number(status.lexicon_entries)} prior node/type pairs`;
    $("#footer-status").textContent = modelsReady ? `Ready · models loaded · ${memory}` : `Ready · ${memory}`;
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
$("#model-select").addEventListener("change", event => {
  state.llmModel = event.target.value;
});
const dropzone = $("#dropzone");
["dragenter","dragover"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave","drop"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));
$("#run-button").addEventListener("click", () => {
  const form = new FormData();
  form.append("paper", state.file);
  form.append("model", state.llmModel);
  begin("/api/extract", {method:"POST", body:form});
});
$("#sample-button").addEventListener("click", () => begin(`/api/extract/sample?model=${encodeURIComponent(state.llmModel)}`, {method:"POST"}));
$("#epanet-sample-button").addEventListener("click", () => begin(`/api/extract/sample/epanet-agentic?model=${encodeURIComponent(state.llmModel)}`, {method:"POST"}));
$("#copy-graph-query").addEventListener("click", async () => {
  if (!state.graphQuery) return;
  await navigator.clipboard.writeText(state.graphQuery);
  toast("Neo4j query copied. Open Neo4j and press Ctrl+V, then Ctrl+Enter.");
});
$("#add-to-graph").addEventListener("click", () => graphAction("add"));
$("#remove-from-graph").addEventListener("click", () => graphAction("remove"));
$("#reset-graph-button").addEventListener("click", () => {
  $("#reset-graph-dialog").showModal();
  $("#reset-confirmation").focus();
});
$("#reset-confirmation").addEventListener("input", event => {
  $("#confirm-graph-reset").disabled = event.target.value !== "RESET";
});
$("#confirm-graph-reset").addEventListener("click", resetSharedGraph);
$("#cancel-graph-reset").addEventListener("click", closeResetDialog);
$("#reset-graph-dialog").addEventListener("cancel", event => {
  event.preventDefault();
  closeResetDialog();
});
$("#history-select").addEventListener("change", async event => {
  if (!event.target.value) return;
  state.jobId = event.target.value;
  state.job = await api(`/api/jobs/${state.jobId}`);
  updateRun(state.job);
  if (state.job.status === "complete") renderResults(state.job);
  else {
    $("#results").hidden = true;
    pollJob();
  }
});
$$(".tab").forEach(tab => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
$$(".relation-filter").forEach(button => button.addEventListener("click", () => {
  state.relationFilter = button.dataset.relationFilter;
  $$(".relation-filter").forEach(item => item.classList.toggle("active", item === button));
  renderRelationships(state.relationshipCandidates || []);
}));
init();
