"""
Dynamic PDF report generation using ReportLab.

Includes realistic transaction details, model performance metrics,
SHAP explainability breakdown, and technical dataset transparency notes.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models.prediction import Prediction


def build_prediction_report(prediction: Prediction, model_metrics: dict | None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=1.5 * cm, rightMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=4, textColor=colors.HexColor("#1e293b"))
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=12)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0f172a"))
    normal = ParagraphStyle("NormalStyle", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#334155"))

    story = []

    story.append(Paragraph("TrustCheck — Smart Transaction Risk Analysis", title_style))
    story.append(Paragraph(f"Transaction Risk Assessment Report | ID: {prediction.id}", subtitle_style))
    story.append(Spacer(1, 4))

    # --- Section 1: Transaction Details ---
    story.append(Paragraph("1. Transaction Information", heading_style))
    amt_str = f"${prediction.amount:,.2f}" if prediction.amount is not None else "N/A"
    tx_time_str = prediction.transaction_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if prediction.transaction_timestamp else prediction.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    tx_data = [
        ["Transaction Amount", amt_str, "Transaction Date/Time", tx_time_str],
        ["Transaction Type", str(prediction.transaction_type or "Online").title(), "Merchant Category", str(prediction.merchant_category or "Shopping").title()],
        ["Location", str(prediction.location or "N/A"), "Recent Activity", str(prediction.derived_features.get("tx_velocity_5m", 0) if prediction.derived_features else "0")],
        ["Card Present", "Yes" if prediction.card_present else "No", "International", "Yes" if prediction.international_transaction else "No"],
    ]
    tx_table = Table(tx_data, colWidths=[4.2 * cm, 4.8 * cm, 4.2 * cm, 4.8 * cm])
    tx_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tx_table)
    story.append(Spacer(1, 6))

    # --- Section 2: Prediction Verdict & Risk Score ---
    shap_data = prediction.shap_explanation or {}
    decision_layer = shap_data.get("decision_layer") or {}
    tc_assessment = decision_layer.get("trustcheck_assessment", "FRAUD" if prediction.is_fraud else "LEGITIMATE")
    verdict = f"{tc_assessment} — {'High Risk / Potential Fraud' if prediction.is_fraud else 'Normal Transaction Activity'}"
    verdict_color = colors.HexColor("#dc2626") if prediction.is_fraud else colors.HexColor("#16a34a")

    story.append(Paragraph("2. TrustCheck Final Assessment & ML Model Evaluation", heading_style))
    summary_data = [
        ["TrustCheck Final Assessment", verdict],
        ["TrustCheck Risk Score", f"{prediction.risk_score:.1f} / 100  ({prediction.risk_level} RISK)"],
        ["ML Model Prediction", f"{decision_layer.get('ml_prediction', 'LEGITIMATE')} (Fraud Prob: {prediction.fraud_probability * 100:.2f}%)"],
        ["ML Legitimate Probability", f"{(1.0 - prediction.fraud_probability) * 100:.2f}%"],
        ["Ensemble Primary Model", prediction.model_name],
        ["Audit Evaluation Time", prediction.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]
    table = Table(summary_data, colWidths=[6.5 * cm, 11.5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (1, 0), (1, 0), verdict_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))

    # --- Section 3: Model performance summary ---
    if model_metrics:
        story.append(Paragraph(f"3. Model Benchmark Validation ({prediction.model_name})", heading_style))
        perf_data = [
            ["Metric", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "MCC"],
            [
                "Test Score",
                f"{model_metrics.get('precision', 0):.4f}",
                f"{model_metrics.get('recall', 0):.4f}",
                f"{model_metrics.get('f1_score', 0):.4f}",
                f"{model_metrics.get('roc_auc', 0):.4f}",
                f"{model_metrics.get('pr_auc', 0):.4f}",
                f"{model_metrics.get('mcc', 0):.4f}",
            ],
        ]
        perf_table = Table(perf_data, colWidths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
        perf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 6))

    # --- Section 4: Explainable AI (SHAP) ---
    story.append(Paragraph("4. Explainable AI: Feature Risk Attribution (SHAP)", heading_style))
    shap_data = prediction.shap_explanation or {}
    fraud_contribs = shap_data.get("top_fraud_contributors", [])
    legit_contribs = shap_data.get("top_legitimate_contributors", [])

    story.append(Paragraph("<b>Top Risk Accelerators (Pushed toward Fraud):</b>", normal))
    if fraud_contribs:
        rows = [["Feature", "Observed Value", "SHAP Impact", "Explanation"]]
        for c in fraud_contribs:
            rows.append([
                c.get("label", c["feature"]),
                f"{c['value']:.3f}",
                f"+{c['shap_value']:.4f}",
                c.get("description", "Positive fraud impact"),
            ])
        t = Table(rows, colWidths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 8.5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc2626")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No significant fraud drivers detected.", normal))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Top Trust Accelerators (Supported Legitimacy):</b>", normal))
    if legit_contribs:
        rows = [["Feature", "Observed Value", "SHAP Impact", "Explanation"]]
        for c in legit_contribs:
            rows.append([
                c.get("label", c["feature"]),
                f"{c['value']:.3f}",
                f"{c['shap_value']:.4f}",
                c.get("description", "Supported legitimacy"),
            ])
        t = Table(rows, colWidths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 8.5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a34a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No significant legitimacy-supporting drivers detected.", normal))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Technical Notice & Dataset Transparency:</b> This system utilizes dynamic feature engineering derived "
        "from transaction amounts, elapsed time windows, and velocity statistics. Realistic transaction metadata (merchant, location, device) "
        "is captured for audit and compliance logging. Underlying ML models are trained with data-leakage-safe pipelines and SMOTE on training splits.",
        ParagraphStyle("Footnote", parent=normal, fontSize=7, leading=9, textColor=colors.HexColor("#64748b")),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
