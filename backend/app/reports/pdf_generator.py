"""
Dynamic PDF report generation using ReportLab.

Every value written into the PDF comes from an actual Prediction DB record
(itself produced by the real ML pipeline) - nothing here is a static
template with placeholder values.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.models.prediction import Prediction


def build_prediction_report(prediction: Prediction, model_metrics: dict | None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=6)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    normal = styles["Normal"]

    story = []

    story.append(Paragraph("Intelligent Credit Card Fraud Detection System", title_style))
    story.append(Paragraph("Automated Fraud Analysis Report", styles["Heading3"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            normal,
        )
    )
    story.append(Spacer(1, 12))

    # --- Prediction summary ---
    verdict = "FRAUD" if prediction.is_fraud else "NOT FRAUD"
    verdict_color = colors.HexColor("#c0392b") if prediction.is_fraud else colors.HexColor("#1e8449")

    story.append(Paragraph("Transaction Prediction", heading_style))
    summary_data = [
        ["Prediction", verdict],
        ["Fraud Probability", f"{prediction.fraud_probability * 100:.2f}%"],
        ["Legitimate Probability", f"{(1 - prediction.fraud_probability) * 100:.2f}%"],
        ["Fraud Risk Score", f"{prediction.risk_score:.0f} / 100"],
        ["Risk Level", prediction.risk_level],
        ["Model Used", prediction.model_name],
        ["Prediction Timestamp", prediction.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]
    table = Table(summary_data, colWidths=[6 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ("TEXTCOLOR", (1, 0), (1, 0), verdict_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    # --- Model performance summary ---
    if model_metrics:
        story.append(Paragraph("Model Performance Summary", heading_style))
        perf_data = [
            ["Metric", "Value"],
            ["Precision", f"{model_metrics['precision']:.4f}"],
            ["Recall", f"{model_metrics['recall']:.4f}"],
            ["F1-Score", f"{model_metrics['f1_score']:.4f}"],
            ["ROC-AUC", f"{model_metrics['roc_auc']:.4f}"],
            ["PR-AUC", f"{model_metrics['pr_auc']:.4f}"],
            ["MCC", f"{model_metrics['mcc']:.4f}"],
        ]
        perf_table = Table(perf_data, colWidths=[6 * cm, 8 * cm])
        perf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(perf_table)

    # --- SHAP explanation ---
    story.append(Paragraph("Why Was This Transaction Flagged?", heading_style))
    shap_data = prediction.shap_explanation or {}
    fraud_contribs = shap_data.get("top_fraud_contributors", [])
    legit_contribs = shap_data.get("top_legitimate_contributors", [])

    story.append(Paragraph("Top features increasing fraud probability:", styles["Heading4"]))
    if fraud_contribs:
        rows = [["Feature", "Value", "SHAP Contribution"]]
        for c in fraud_contribs:
            rows.append([c["feature"], f"{c['value']:.4f}", f"+{c['shap_value']:.4f}"])
        t = Table(rows, colWidths=[4 * cm, 5 * cm, 5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No significant fraud-pushing features detected.", normal))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Top features decreasing fraud probability:", styles["Heading4"]))
    if legit_contribs:
        rows = [["Feature", "Value", "SHAP Contribution"]]
        for c in legit_contribs:
            rows.append([c["feature"], f"{c['value']:.4f}", f"{c['shap_value']:.4f}"])
        t = Table(rows, colWidths=[4 * cm, 5 * cm, 5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e8449")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No significant legitimacy-supporting features detected.", normal))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Note: This report is generated by an academic prototype and SHAP values reflect the "
        "trained model's learned behaviour on anonymized PCA features (V1-V28); they are not "
        "directly interpretable as real-world transaction attributes.",
        ParagraphStyle("Footnote", parent=normal, fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
