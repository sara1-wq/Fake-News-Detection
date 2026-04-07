import sys
sys.path.insert(0, r'C:\Users\Admin\OneDrive\Desktop\.vscode\tt.js\python')
from predict import predict_news

cases = [
    'Federal Reserve Raises Interest Rates - The Federal Reserve announced a quarter-point increase in interest rates to combat inflation.',
    'Aliens Spotted Hovering Over Major Cities - Multiple witnesses reported seeing UFOs above New York and Los Angeles. Government has no comment.',
    'Employment Rate Reaches Five-Year High - New employment statistics show job creation exceeded expectations for the third consecutive month.',
    "Miracle Cure Discovered That Doctors Don't Want You to Know About - A secret compound can cure every disease known to mankind but big pharma is suppressing it."
]

for c in cases:
    label, conf = predict_news(c)
    print(f'{c[:70]:<70} ... {label} ({conf:.3f})')
