from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BLUE      = colors.HexColor('#1a3a6b')
LIGHTBLUE = colors.HexColor('#e8edf7')
ACCENT    = colors.HexColor('#2c5aa0')
GRAY      = colors.HexColor('#666666')
LIGHTGRAY = colors.HexColor('#f5f5f5')
WHITE     = colors.white

doc = SimpleDocTemplate(
    'results/summary_for_dr_liu.pdf',
    pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.85*inch, bottomMargin=0.85*inch,
    title='VIX Regime Classification - Research Summary',
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle('title',
    fontName='Helvetica-Bold', fontSize=20, textColor=BLUE,
    spaceAfter=4, leading=24)

subtitle_style = ParagraphStyle('subtitle',
    fontName='Helvetica', fontSize=10, textColor=GRAY,
    spaceAfter=16, leading=14)

h2_style = ParagraphStyle('h2',
    fontName='Helvetica-Bold', fontSize=12, textColor=ACCENT,
    spaceBefore=14, spaceAfter=5, leading=16)

body_style = ParagraphStyle('body',
    fontName='Helvetica', fontSize=9.5, textColor=colors.HexColor('#222222'),
    spaceAfter=6, leading=14)

caption_style = ParagraphStyle('caption',
    fontName='Helvetica-Oblique', fontSize=8.5, textColor=GRAY,
    spaceAfter=8, leading=12)

def section(title):
    return [
        Paragraph(title, h2_style),
        HRFlowable(width='100%', thickness=0.5, color=ACCENT, spaceAfter=5),
    ]

def tbl(headers, rows, col_widths):
    data = [headers] + rows
    t = Table(data, colWidths=[w*inch for w in col_widths])
    style = TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  8.5),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,1), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHTGRAY]),
        ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING',(0,0), (-1,-1), 6),
    ])
    t.setStyle(style)
    return t

story = []

story.append(Paragraph('VIX Regime Classification', title_style))
story.append(Paragraph('Research Summary', title_style))
story.append(Paragraph(
    'Akshat Gupta &nbsp;&nbsp;|&nbsp;&nbsp; TAMS, University of North Texas &nbsp;&nbsp;|&nbsp;&nbsp; '
    'Advisor: Prof. Jianguo Liu &nbsp;&nbsp;|&nbsp;&nbsp; May 2026',
    subtitle_style))
story.append(HRFlowable(width='100%', thickness=1.5, color=BLUE, spaceAfter=12))

story += section('Overview')
story.append(Paragraph(
    'This project applies the selective predicting methodology from Liu &amp; Jiang (2020) to VIX '
    'volatility regime classification. The goal is to predict whether the VIX index will be in a '
    'high-volatility regime (VIX &gt;= 20) N trading days ahead, and to show that confidence-based '
    'selective predicting consistently improves accuracy over baseline models.', body_style))

story += section('Data &amp; Features')
story.append(tbl(
    ['Source', 'Series', 'Features', 'Dataset'],
    [['Yahoo Finance + FRED', 'VIX, VIX3M, SP500, Gold, TNY, DXY, SKEW',
      '36 engineered', '~5,000 days (1990-present)']],
    [1.4, 2.8, 1.2, 1.4]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    'Features include VIX technicals (moving averages, Bollinger bands, RSI, momentum), '
    'cross-asset signals, S&amp;P 500 equity indicators, and macro factors (Fed Funds Rate, yield curve). '
    'Chronological 80/20 train/test split with no shuffling.', body_style))

story += section('Models &amp; Methodology')
story.append(Paragraph(
    '<b>Random Forest</b> and <b>XGBoost</b> (HistGradientBoostingClassifier) with hyperparameter '
    'tuning via RandomizedSearchCV (n_iter=50, TimeSeriesSplit). Both use isotonic calibration '
    '(CalibratedClassifierCV, 5-fold TimeSeriesSplit) for well-calibrated probability estimates.',
    body_style))
story.append(Paragraph(
    '<b>Selective Predicting:</b> A prediction is made only when model confidence exceeds threshold t '
    '(i.e. P &gt; 0.5 + t or P &lt; 0.5 - t). Uncertain days near 0.5 are abstained from, '
    'trading coverage for higher accuracy.', body_style))

story += section('Main Results  (VIX &gt;= 20, t = 0.25)')
story.append(tbl(
    ['N (days ahead)', 'RF Baseline', 'RF Selective', 'Gain', 'Coverage'],
    [['5',  '83.8%', '91.1%', '+7.3%',  '78.4%'],
     ['10', '78.6%', '88.8%', '+10.2%', '70.5%'],
     ['15', '74.9%', '90.0%', '+15.1%', '58.1%'],
     ['20', '72.6%', '90.9%', '+18.3%', '50.6%'],
     ['25', '69.7%', '91.1%', '+21.5%', '32.9%']],
    [1.4, 1.3, 1.3, 1.0, 1.0]))
story.append(Spacer(1, 6))
story.append(tbl(
    ['N (days ahead)', 'XGB Baseline', 'XGB Selective', 'Gain', 'Coverage'],
    [['5',  '82.9%', '90.7%', '+7.8%',  '79.4%'],
     ['10', '79.0%', '91.0%', '+12.0%', '61.1%'],
     ['15', '73.5%', '87.7%', '+14.2%', '54.6%'],
     ['20', '74.4%', '92.1%', '+17.7%', '35.4%'],
     ['25', '66.5%', '92.5%', '+26.0%', '24.1%']],
    [1.4, 1.3, 1.3, 1.0, 1.0]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    'Confidence-based selective predicting yields consistent accuracy gains across all prediction '
    'horizons. Gains increase with N - at 25 days ahead, XGBoost improves from 66.5% to 92.5%.',
    caption_style))

story += section('Robustness - Multiple VIX Thresholds (RF Accuracy Gain)')
story.append(tbl(
    ['N', 'VIX >= 18', 'VIX >= 20', 'VIX >= 22'],
    [['5',  '+7.6%',  '+9.0%',  '+8.9%'],
     ['10', '+11.1%', '+8.9%',  '+10.7%'],
     ['15', '+16.6%', '+10.8%', '+11.4%'],
     ['20', '+15.2%', '+15.1%', '+13.1%'],
     ['25', '+25.9%', '+8.9%',  '+9.1%']],
    [0.8, 1.5, 1.5, 1.5]))
story.append(Paragraph('All gains are positive across every VIX threshold. Result is robust.', caption_style))

story += section('Sustained Regime Labels')
story.append(Paragraph(
    'A harder variant: predict whether VIX &gt;= 20 for <b>3 consecutive days</b> starting at day N. '
    'Models perform <i>better</i> on this task - sustained regimes have stronger precursors.',
    body_style))
story.append(tbl(
    ['N', 'RF Sustained Selective', 'RF Instantaneous Selective'],
    [['5','94.0%','91.1%'],['10','93.8%','88.8%'],
     ['15','93.8%','90.0%'],['20','94.3%','90.9%'],['25','89.9%','91.1%']],
    [0.8, 2.6, 2.6]))

story += section('Walk-Forward Validation (5 Expanding Windows)')
story.append(tbl(
    ['Model', 'N', 'Mean Baseline', 'Mean Selective', 'Delta'],
    [['Random Forest', '5',  '82.8% +/- 9.7%', '92.2% +/- 1.7%', '+9.4%'],
     ['XGBoost',       '5',  '84.0% +/- 8.0%', '84.6% +/- 9.7%', '+0.6%'],
     ['Random Forest', '10', '72.3% +/- 22.7%','85.9% +/- 7.7%', '+13.7%']],
    [1.5, 0.5, 1.6, 1.6, 0.8]))
story.append(Paragraph(
    'RF selective predicting is notably more consistent across time (lower std dev) than the baseline.',
    caption_style))

story += section('Feature Importance (Random Forest, avg across N = 5, 10, 20)')
story.append(tbl(
    ['Rank', 'Feature', 'Avg Importance', 'Category'],
    [['1', 'VIX_SMA10',         '19.8%', 'VIX trend'],
     ['2', 'VIX_SMA20',         '15.3%', 'VIX trend'],
     ['3', 'VIX_BB_UPPER',      '10.9%', 'VIX Bollinger bands'],
     ['4', 'VIX_BB_LOWER',      '10.8%', 'VIX Bollinger bands'],
     ['5', 'SP500_DRAWDOWN',    '6.8%',  'Equity risk'],
     ['6', 'SP500_RVOL',        '5.7%',  'Realized volatility'],
     ['7', 'FEDFUNDS_CHANGE3M', '2.2%',  'Macro'],
     ['8', 'VIX_BB_STD',        '2.2%',  'VIX vol-of-vol']],
    [0.5, 1.8, 1.3, 2.4]))
story.append(Paragraph(
    'VIX trend/level features account for ~57% of predictive power. Macro features grow more '
    'important at longer prediction horizons.',
    caption_style))

story += section('GitHub Repository')
story.append(Paragraph(
    '<font color="#2c5aa0"><u>https://github.com/akshatg10104/VIX_research</u></font>',
    body_style))

doc.build(story)
print("PDF saved -> results/summary_for_dr_liu.pdf")
