from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT = "/Users/Akshat/vix_research/vix final draft/VIX_Research_Summary_June2026.pdf"

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
                        leftMargin=1*inch, rightMargin=1*inch,
                        topMargin=1*inch, bottomMargin=1*inch)

styles = getSampleStyleSheet()

title_style = ParagraphStyle('title', fontSize=14, fontName='Helvetica-Bold',
                              alignment=TA_CENTER, spaceAfter=4)
sub_style  = ParagraphStyle('sub', fontSize=10, fontName='Helvetica',
                              alignment=TA_CENTER, spaceAfter=2)
section_style = ParagraphStyle('section', fontSize=11, fontName='Helvetica-Bold',
                                spaceBefore=14, spaceAfter=4)
body_style = ParagraphStyle('body', fontSize=9.5, fontName='Helvetica',
                              leading=14, spaceAfter=6)
small_style = ParagraphStyle('small', fontSize=8.5, fontName='Helvetica',
                               leading=12, spaceAfter=4)

def section(text):
    return [Paragraph(text, section_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=4)]

def body(text):
    return Paragraph(text, body_style)

def tbl(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#F2F2F2')]),
        ('GRID',      (0,0), (-1,-1), 0.4, colors.grey),
        ('TOPPADDING',(0,0), (-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('LEFTPADDING',(0,0),(-1,-1),5),
    ]))
    return t

story = []

# Title block
story += [
    Paragraph("VIX Regime Classification — Research Summary", title_style),
    Paragraph("Akshat Gupta | TAMS, University of North Texas | Advisor: Prof. Jianguo Liu | June 2026", sub_style),
    HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10),
]

# Overview
story += section("Overview")
story += [body(
    "This summary reflects the current state of the paper as of June 2026. The paper has been significantly "
    "reframed since the May 2026 summary. It began as a positive ML result — showing that selective prediction "
    "achieves 90–93% accuracy on VIX regime classification. Deeper analysis revealed the accuracy is entirely "
    "driven by VIX autocorrelation, not genuine forecasting skill. The paper is now a diagnostic decomposition "
    "that formally identifies and tests this 'persistence illusion' through a six-component protocol, replicated "
    "across three tasks: VIX regimes, S&P 500 trend regimes, and yield curve inversion."
)]

# What changed
story += section("Key Changes Since May 2026 Summary")
story += [body(
    "<b>Pipeline:</b> RandomizedSearchCV hyperparameter tuning removed. All results use a fixed-spec calibrated "
    "pipeline: RF (n_estimators=200, max_depth=10, min_samples_leaf=5) and HGB (max_iter=200, max_depth=5, "
    "lr=0.1), both with isotonic calibration via CalibratedClassifierCV (5-fold TimeSeriesSplit). "
    "XGBoost replaced by HistGradientBoosting (HGB) throughout."
), body(
    "<b>Framing:</b> Positive result framing removed. The headline accuracy (90–93%) is shown to be an artifact "
    "of VIX autocorrelation. A 3-feature logistic regression matches the 36-feature RF at 4 of 5 horizons. "
    "No RF-vs-persistence gap reaches statistical significance under moving-block bootstrap."
), body(
    "<b>New sections added:</b> (1) Regime transition analysis — 0% accuracy on calm→high spike days at every "
    "horizon, even with cost-weighting up to 20×. (2) S&P 500 and yield curve cross-task replications. "
    "(3) Synthetic positive control confirming the diagnostic has power to detect real signal."
)]

# Main results
story += section("Main Results (VIX ≥ 20, τ = 0.25)")
story += [
    tbl([
        ['N', 'RF Baseline', 'RF Selective', 'RF Gain', 'RF Coverage', 'RF BalAcc', 'HGB Sel', 'HGB Gain', 'HGB Cov'],
        ['5',  '83.4%', '91.9%', '+8.5 pp', '76.3%', '88.4%', '92.1%', '+8.9 pp',  '71.1%'],
        ['10', '79.5%', '90.0%', '+10.5 pp','60.1%', '84.8%', '92.2%', '+13.6 pp', '63.1%'],
        ['15', '76.2%', '90.0%', '+13.8 pp','51.3%', '83.9%', '92.2%', '+21.3 pp', '36.0%'],
        ['20', '76.8%', '92.7%', '+15.9 pp','31.7%', '79.1%', '94.8%', '+22.2 pp', '26.8%'],
        ['25', '73.4%', '90.3%', '+16.9 pp','26.9%', '50.0%*','92.7%', '+26.5 pp', '19.2%'],
    ], col_widths=[0.3*inch,0.8*inch,0.8*inch,0.7*inch,0.8*inch,0.7*inch,0.7*inch,0.8*inch,0.7*inch]),
    Spacer(1, 4),
    Paragraph("* At N=25 balanced accuracy = 50% (coin flip). Model predicts calm on every covered day.", small_style),
]

# Baselines
story += section("Comparison to Naive Baselines (New)")
story += [body(
    "Central finding: a coverage-matched persistence rule and a 3-feature HAR logistic regression both match "
    "the 36-feature RF. No gap is statistically significant."
),
    tbl([
        ['N', 'Coverage', 'Persistence', 'HAR-LR (3 feat)', 'RF Selective', 'RF minus Persist'],
        ['5',  '76.3%', '91.6%', '91.6%', '91.9%', '+0.3 pp'],
        ['10', '60.1%', '88.1%', '91.1%', '90.0%', '+1.9 pp'],
        ['15', '51.3%', '87.9%', '90.2%', '90.0%', '+2.1 pp'],
        ['20', '31.7%', '90.5%', '94.2%', '92.7%', '+2.3 pp'],
        ['25', '26.9%', '83.8%', '92.0%', '90.3%', '+6.5 pp'],
    ], col_widths=[0.3*inch, 0.75*inch, 0.85*inch, 1.0*inch, 0.85*inch, 1.0*inch]),
    Spacer(1, 4),
    Paragraph(
        "At N≥10, RF and persistence make identical predictions on every jointly covered day (McNemar b=c=0). "
        "The accuracy gap is entirely explained by which days each rule covers, not forecasting skill. "
        "Bootstrap 95% CI at N=10: RF vs. persistence [−1.11, +4.17]; HAR vs. RF [−1.08, +4.18].",
        small_style),
]

# Persistence quantification
story += section("Persistence Quantification (New)")
story += [
    tbl([
        ['N', 'P(High | High today)', 'P(High | Calm today)', 'Persistence Ratio', 'Same-Regime % of RF Correct'],
        ['5',  '84.2%', '8.6%',  '9.8×', '99.0%'],
        ['10', '79.5%', '11.2%', '7.1×', '99.6%'],
        ['15', '75.9%', '13.2%', '5.8×', '99.8%'],
        ['20', '73.1%', '14.8%', '4.9×', '100.0%'],
        ['25', '70.0%', '16.5%', '4.2×', '100.0%'],
    ], col_widths=[0.3*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.5*inch]),
    Spacer(1, 4),
    Paragraph("At N≥20, 100% of RF correct predictions are same-regime calls. The model adds no directional forecasting beyond VIX autocorrelation.", small_style),
]

# Transition analysis
story += section("Regime Transition Analysis (New)")
story += [body(
    "The model achieves 0% accuracy on calm→high spike days at every horizon. The confidence filter covers "
    "87% of calm-persisting days but systematically abstains or makes wrong predictions on spike days. "
    "Cost-weighting up to 20× does not fix this — the failure is informational, not objective-function-driven."
)]

# Cross-task replications
story += section("Cross-Task Replications (New)")
story += [
    tbl([
        ['Task', 'Persistence Ratio (N=5)', 'Transition Acc', 'RF minus Persistence'],
        ['VIX regime',           '9.8×',  '0%',   '+0.3 pp (not significant)'],
        ['S&P 500 trend regime', '23.8×', '0%',   '−4.0 pp (persistence wins)'],
        ['Yield curve inversion','75.5×', '0%',   '−53.0 pp (persistence wins)'],
        ['Synthetic (oracle injected)', '4.4×', '54.9%', '+8.7 pp (genuine skill detected)'],
    ], col_widths=[1.5*inch, 1.3*inch, 0.9*inch, 1.8*inch]),
    Spacer(1, 4),
    Paragraph(
        "Severity of the illusion scales with the persistence ratio. The synthetic positive control confirms "
        "the diagnostic has power: when a real predictive signal is injected, the protocol correctly identifies it.",
        small_style),
]

# Feature importance
story += section("Feature Importance (SHAP, New)")
story += [body(
    "VIX technical features (moving averages, Bollinger bands) account for 71.9% of model SHAP mass at N=5 "
    "and 64.7% at N=10. S&P 500 features are a secondary share (~19–21%). Macro and cross-asset features "
    "account for under 10% and are negligible. The model overwhelmingly learns VIX autocorrelation."
)]

# Status
story += section("Current Status")
story += [body(
    "Paper is 53 pages double-spaced. All results are complete and consistent across three replication tasks. "
    "Goal is conference submission (ICAIF, QuantFin, or similar ML-finance venue). "
    "Next step: identify target venue and cut to submission length."
)]

doc.build(story)
print(f"Saved: {OUTPUT}")
