# Prompt for LLM (Code Specialist)

You are a specialist LLM designed to implement mathematical software systems.
You are now tasked with developing a multi-agent AI system that creates a
central economic plan based on cybernetic principles, using Input-Output
analysis and labor-time accounting.

## Task Objective

Build a system that accepts economic input-output tables and produces a
markdown report of a 5-year central plan, optimizing for labor-time and
physical constraints. Use the following steps and models:

---

## Step 1: Matrix Construction

Parse the following data (provided separately):

- **Technology Matrix** \( A \in \mathbb{R}^{n \times n} \)
- **Final Demand Vector** \( d \in \mathbb{R}^{n \times 1} \)
- **Labor Input Vector** \( l \in \mathbb{R}^{1 \times n} \)
- **Resource Matrix** \( R \in \mathbb{R}^{m \times n} \)
- **Max Resources** \( R_{max} \in \mathbb{R}^{m \times 1} \)

Ensure:

- \( \rho(A) < 1 \)
- Matrices are dimensionally consistent

---

## Step 2: Compute Total Output (Year 1)

Use the Leontief model:
\[
x = (I - A)^{-1} d
\]

Validate:

- \( x \geq 0 \)

---

## Step 3: Compute Labor Values

\[
v = l (I - A)^{-1}
\]
\[
L = v \cdot d
\]

---

## Step 4: Solve LP to Minimize Labor

Formulate the LP:

**Objective**:
\[
\min l \cdot x
\]

**Subject to**:

- \( (I - A)x \ge d \)
- \( Rx \le R_{max} \)
- \( x \ge 0 \)

Use `cvxpy` or `scipy.optimize.linprog`.

---

## Step 5: Project Years 2–5

For each year \( t \in [2,5] \):

- Update \( A_t \), \( l_t \) based on investment decisions
- Apply:
  \[
  x_t = A_t x_t + d_{c,t} + d_{i,t}
  \]
  \[
  K_{t+1} = (I - \delta)K_t + M d_{i,t}
  \]

---

## Step 6: Multi-Agent AI

Implement agents:

- `ManagerAgent`: coordinates loop
- `EconomicsAgent`: runs \( \partial x / \partial A_{ij} \), forecasts \( A_t \)
- `ResourceAgent`: modifies \( R \), checks \( Rx \le R_{max} \)
- `PolicyAgent`: converts natural language goals into updates to \( d \), adds constraints
- `WriterAgent`: generates final markdown

---

## Step 7: Generate Markdown Output

Format report with:

- Executive Summary
- Output vector \( x \)
- Labor table \( v_j, l_j x_j \)
- Resource use \( Rx \)
- Constraint violations
- Sensitivity analysis
- Scoring:
  \[
  \text{Score} = \sum w_k f(P_k, T_k)
  \]

---

## Requirements

- Use modular code
- Document functions with docstrings
- Handle sparse matrices efficiently
- Support override inputs from natural language

---

## Deliverables

- Python source code (modular structure)
- Sample markdown plan for synthetic data
- Unit tests for all modules
