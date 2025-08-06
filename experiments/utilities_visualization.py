from plotly import graph_objects as go
from plotly import io as pio

journal_template = go.layout.Template()
journal_template.layout.font = dict(family="Computer Modern, serif", size=14, color="black")
journal_template.layout.title.font = dict(size=18)
journal_template.layout.xaxis = dict(showgrid=True, gridcolor='lightgray', linecolor='black', showline=True, mirror=True, ticks='outside', ticklen=5)
journal_template.layout.yaxis = dict(showgrid=True, gridcolor='lightgray', linecolor='black', showline=True, mirror=True, ticks='outside', ticklen=5)
journal_template.layout.margin = dict(l=40, r=20, t=60, b=40)
pio.templates['journal_theme'] = journal_template
pio.templates.default = "journal_theme"