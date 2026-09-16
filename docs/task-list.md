1. [Main Agent Supervised](#1-main-agent-supervised)  
2. [SubAgent Collaboration](#2-subagent-collaboration)  
3. [Timeout](#3-timeout)  
4. [Order](#4-order)  
5. [Inconsistency](#5-inconsistency)

> **Status (M0–M5 built; only M6, scale & baseline, remains).** All 15 archetypes
> below — across the 5 top-level categories — are implemented with at least one
> seed. The line currently ships **20 conflict seeds** in
> `clasharness/data/scenario_seeds/conflict_vlm_v1/` and **23 registered conflict
> plugins**, backed by **7 `StatefulProtocolSkill` protocol families** in
> `clasharness/skills/impl/` (approval, booking, containment, deploy, lock,
> pairing, review).
> Archetype-to-category coverage is defined by `ARCHETYPE_CATEGORY` and computed by
> `archetype_coverage()` / `category_coverage()` in
> `clasharness/scenario/conflict_vlm_spec.py`, and is enforced by
> `tests/test_archetype_coverage.py`.

#### **1\. Main Agent Supervised**

* **Incident Commander Selection:**  During a service outage, multiple agents observe different parts of the system, such as frontend errors, backend logs, database status, network health, and recent deployments. The main agent must decide who should lead the incident, who should diagnose, who can execute recovery actions, and who should communicate updates.  
* **Access Request Approval:**  A user requests access to a repository, database, dashboard, or customer dataset. The main agent must coordinate the requester, manager, application owner, security reviewer, and IAM agent. It must prevent unauthorized access and reconcile conflicting approval scopes, such as full access, read-only access, or temporary access.

#### **2\. SubAgent Collaboration**

* **Multi-Party Access Permit:** Access to a sensitive system is granted only when multiple agents provide valid approvals within the same time window. For example, manager approval, data-owner approval, and security approval must all be simultaneously active before the IAM agent can grant access. The main agent must synchronize approval timing, resolve scope conflicts, and prevent partial or expired approvals from triggering access.  
* **Coordinated Multi-Point Containment:** In a cybersecurity incident, multiple defense agents must act together to contain a threat. For example, the IAM agent revokes active sessions, the email-security agent disables suspicious forwarding rules, the endpoint agent isolates a device, and the cloud-security agent blocks compromised tokens. Handling only one control point is insufficient; the main agent must coordinate synchronized containment across agents.  
* **Collaborative Contract Review:** A contract requires review by legal, privacy, security, business, visual-layout, and signing agents. Each agent is responsible for a different aspect of the document. The main agent must assign non-overlapping review roles, integrate their findings, resolve blocking issues, and prevent signing before all required approvals are complete.  
* **Multi-Tab Slot / Stock Monitoring:** The user needs to secure a volatile online resource, such as an appointment slot, ticket, hotel room, class seat, or limited-stock item. Multiple agents monitor different tabs, pages, locations, or candidate options in parallel. When one agent detects availability, the main agent must quickly validate constraints, avoid duplicate or conflicting holds, and trigger a single booking or hold action before the opportunity disappears. Success depends on parallel monitoring, stale-state handling, and coordinated commitment.

#### **3\. Timeout**

* **Service Level Agreement Support Response:** A customer support ticket must receive a first response before an SLA deadline. Some agents can provide fast partial evidence, while a specialist agent may be slow. The main agent must decide whether to send a safe holding response before the deadline while continuing deeper investigation.  
* **Expiring Resource Hold:** A meeting room, flight fare, hotel room, appointment slot, or shopping cart hold is about to expire. The main agent must decide whether to place a reversible hold while waiting for participant, payer, or policy approval, without prematurely performing an irreversible action such as final booking or payment.  
* **Multi-Device Setup / Pairing:** The user needs to set up or pair devices across multiple interfaces, such as linking a phone with a laptop, connecting a smart-home device, configuring a router, or migrating an account across devices. Multiple agents observe and operate different device screens. The task requires synchronized actions, such as entering a pairing code, approving a prompt, scanning a QR code, or confirming a login request within a short time window. Success depends on coordinating both endpoints before the session expires.
  * **Reproducible design:** The pairing process can be simulated with deterministic device-state machines. For example, one device displays a time-limited pairing code, while the other device must enter or confirm the same code before expiration. The sandbox can reliably control code lifetime, prompt timing, mismatch cases, and session reset behavior, making the task reproducible and automatically evaluable.

#### **4\. Order**

* **Employee Onboarding Workflow**  
  A new employee onboarding process requires strict prerequisites: identity verification, account creation, permission assignment, device enrollment, and orientation scheduling. The main agent must enforce the correct order and block downstream actions when earlier steps are incomplete or inconsistent.  
* **IT Change / Deployment Gate**  
  A software hotfix, configuration change, or database migration must follow a controlled deployment workflow: change request, risk assessment, approval, deployment window, deployment, and monitoring. The main agent must prevent premature deployment, handle emergency approval when needed, and ensure post-deployment monitoring.

#### **5\. Inconsistency**

* **Expense Reimbursement Conflict**  
  A reimbursement request contains conflicting evidence across receipt images, bank transactions, reimbursement policy, manager approval, and finance compliance. The main agent must determine the valid reimbursable amount, distinguish actual charge from allowed reimbursement, and prevent incorrect form submission.  
* **Distributed Lock for Shared Resource**  
  Multiple agents need to edit or update the same shared resource, such as a contract, spreadsheet, ticket record, or project document. The main agent must acquire and release edit locks, assign non-overlapping sections, prevent overwrite conflicts, and verify the final merged state. This tests whether the main agent can avoid lost updates and inconsistent shared-state modifications.  
* **Deadlock in Multi-Approval Workflow**  
  A task requires approvals from multiple agents, but each agent is waiting for another agent’s confirmation before proceeding. For example, Legal waits for Security approval, Security waits for Business justification, and Business waits for Legal risk assessment. The main agent must detect the deadlock, break the circular dependency, request provisional decisions, or escalate to a higher-level approver.  
* **Barrier Synchronization for Deployment**  
  A deployment, rollback, database migration, or traffic cutover can only proceed when multiple agents are simultaneously ready. For example, the database agent, deployment agent, traffic-routing agent, monitoring agent, and service owner must all reach a ready state before the main agent triggers the cutover. This tests whether the main agent can enforce a synchronization barrier and prevent premature or partially coordinated execution.  
  * **Reproducible simulation:** The sandbox implements the CI/CD workflow as a deterministic state machine with separate readiness gates for app deployment, database migration, traffic routing, monitoring, and service-owner approval. Each gate can be set to pending, ready, failed, stale, or expired, allowing controlled reproduction of synchronization failures.  
  * **Evaluation trigger:** The cutover action succeeds only when all readiness gates are simultaneously valid. Premature cutover, stale readiness, missing approval, or degraded monitoring are automatically detected as failure cases.  
    

   