# Offline local tutor transport

The retained learning campaign uses installed local tutor weights through **`http://127.0.0.1:11434`**. Its learner/tutor API path does not use an internet endpoint, cloud model, model download or package installation. The current pinned teacher is `ministral-3:3b`, digest `f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`.

The executable path is:

1. [home_learning.resume](../experiments/home_learning.py) authenticates the retained session and calls the existing campaign in the same Python process.
2. [continuous_tutor_campaign_v3](../experiments/continuous_tutor_campaign_v3.py) calls [verified_tutor_author_v2.author_curriculum](../experiments/verified_tutor_author_v2.py) for a pending authorship stage. Previously committed decisions are reused on resume.
3. The author uses direct `http.client.HTTPConnection` with the numeric host and port fixed in [curriculum_tutor.py](../brain_in_computer/curriculum_tutor.py). Its route allowlist is `/api/tags`, `/api/show` and `/api/chat`. It does not consult proxy or `OLLAMA_HOST` environment settings, follow redirects, resolve a supplied hostname, or expose a pull/download route.
4. Before generation, `/api/tags` must identify exactly the requested installed model and caller-pinned digest. `/api/show` must return local model details. Both reject nonempty `remote_host` or `remote_model` metadata; model names containing `cloud` are refused. The digest is checked again after generation. Each durable request allows one chat attempt, records actual usage and uses `keep_alive=0`.
5. Native practice and testing run in a separate [continuous_tutor_worker](../experiments/continuous_tutor_worker.py) process through the same Python executable, a fixed module and the project directory. The worker does not contact a teacher. Local PowerShell process inventory is used to check for competing learning jobs; neither launcher invokes an installer or downloader.

A missing service, missing model or failed pre-generation identity check produces an explicitly recorded procedural fallback. An uncertain generation is preserved without an automatic retry. This is not evidence of a successful tutor-assisted cycle. The tutor chooses bounded lesson recipes; independent curriculum code owns facts and labels, and native testing disconnects the tutor. Arbitrary English teaching and a useful tutor advantage remain separate research questions.

Use the existing virtual environment and already installed model; setup and resume instructions are in [HOME_CURRENT_LEARNING.md](HOME_CURRENT_LEARNING.md). Internet access is unnecessary for this API path once those local prerequisites exist, so Wi-Fi/Ethernet can be disconnected without changing the learning design. Learning does not automatically download missing prerequisites.

This is a source audit of the learning path, not an OS air gap or a guarantee about unrelated Ollama desktop/app traffic. The separately running local service and other applications retain their own settings. No firewall, network configuration or machine setting was changed by this audit. Actual service/model preflight and learning receipts are recorded separately; this document itself does not claim a new training run.
