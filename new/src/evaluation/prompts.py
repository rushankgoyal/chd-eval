"""
CHD-CXR Prompting Strategies
=============================
Four prompting paradigms for CHD classification from pediatric chest X-rays.
Each value in :data:`PROMPTS` is a ready-to-use prompt string that works with
any VLM accepting text + image input.

Paradigms
---------
ZSD  – Zero-Shot Direct: minimal framing, forces a bare label response.
RCE  – Role-Conditioned Expert: 20-year pediatric cardiologist persona.
CoT  – Chain-of-Thought Medical Reasoning: 5-step pathophysiological framework.
         Parser looks for the terminal "DIAGNOSIS: <label>" line.
RAC  – Reference-Anchored Classification: canonical radiographic features per class.
"""

from __future__ import annotations

PROMPTS: dict[str, str] = {
    # ------------------------------------------------------------------
    # P1 – Zero-Shot Direct (ZSD)
    # Baseline: minimal framing, forces a bare label response.
    # ------------------------------------------------------------------
    "ZSD": (
        "You are a medical image analysis system. "
        "Examine this pediatric chest X-ray and classify it into exactly one of the "
        "following four categories:\n"
        "  • ASD (Atrial Septal Defect)\n"
        "  • VSD (Ventricular Septal Defect)\n"
        "  • PDA (Patent Ductus Arteriosus)\n"
        "  • Normal\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),

    # ------------------------------------------------------------------
    # P2 – Role-Conditioned Expert (RCE)
    # Expert persona conditioning; expected to boost accuracy ~8–12%.
    # ------------------------------------------------------------------
    "RCE": (
        "You are an expert pediatric cardiologist and radiologist with 20 years of "
        "experience interpreting neonatal and pediatric chest X-rays. "
        "You are reviewing a chest radiograph from a pediatric patient for possible "
        "congenital heart disease (CHD).\n\n"
        "Classify this image into exactly one of:\n"
        "  • ASD (Atrial Septal Defect)\n"
        "  • VSD (Ventricular Septal Defect)\n"
        "  • PDA (Patent Ductus Arteriosus)\n"
        "  • Normal\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),

    # ------------------------------------------------------------------
    # P3 – Chain-of-Thought Medical Reasoning (CoT)
    # Step-by-step pathophysiological reasoning before final diagnosis.
    # Parser looks for the terminal "DIAGNOSIS: <label>" line.
    # ------------------------------------------------------------------
    "CoT": (
        "You are an expert pediatric cardiologist. "
        "Examine this chest X-ray step by step using the following framework:\n\n"
        "Step 1 – Cardiac silhouette: Assess size (cardiothoracic ratio), shape, and borders.\n"
        "Step 2 – Pulmonary vascularity: Describe pulmonary blood flow "
        "(increased, normal, or decreased).\n"
        "Step 3 – Mediastinal structures: Evaluate the aortic knuckle, superior "
        "mediastinum width, and tracheal position.\n"
        "Step 4 – Bony and parenchymal findings: Note rib notching, lung fields, "
        "and pleural spaces.\n"
        "Step 5 – Integration: Synthesise findings into a single diagnosis.\n\n"
        "After your reasoning, output your final answer on the last line in the exact format:\n"
        "DIAGNOSIS: <label>\n"
        "where <label> is one of: ASD, VSD, PDA, Normal."
    ),

    # ------------------------------------------------------------------
    # P4 – Reference-Anchored Classification (RAC)
    # Canonical per-class radiographic features provided as textual anchors.
    # ------------------------------------------------------------------
    "RAC": (
        "You are an expert pediatric cardiologist. "
        "Classify this pediatric chest X-ray into one of the four categories below, "
        "using the canonical radiographic features as anchors:\n\n"
        "ASD (Atrial Septal Defect):\n"
        "  – Mild to moderate cardiomegaly with right heart enlargement\n"
        "  – Increased pulmonary vascularity (shunt vascularity / plethora)\n"
        "  – Prominent main pulmonary artery segment\n\n"
        "VSD (Ventricular Septal Defect):\n"
        "  – Cardiomegaly proportional to shunt size\n"
        "  – Increased pulmonary vascularity (biventricular volume overload pattern)\n"
        "  – Enlarged left atrium and left ventricle\n\n"
        "PDA (Patent Ductus Arteriosus):\n"
        "  – Cardiomegaly with left heart prominence\n"
        "  – Pulmonary oedema or markedly increased pulmonary flow\n"
        "  – Prominent aortic arch / knuckle\n\n"
        "Normal:\n"
        "  – Normal cardiothoracic ratio (< 0.5)\n"
        "  – Normal pulmonary vascularity\n"
        "  – Clear lung fields, no mediastinal widening\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),
}
