"""
Test the fake news examples against the main BERT model.
"""
import sys
from predict import predict_news # Import the main prediction function

# Test fake news examples
fake_news_examples = [
    "Government Implants Tracking Chips in Driver's Licenses - New evidence shows authorities secretly embedding RFID chips to monitor citizens' locations.",
    "Elite Secret Society Controls All World Banks - Leaked documents reveal the Illuminati runs every major financial institution globally.",
    "5G Towers Cause Instant Brain Damage - Scientists confirm exposure to 5G radiation causes permanent neurological damage within hours.",
    "Doctors Hate This One Weird Trick to Never Get Sick Again - Eat this one simple fruit and you'll never need medicine again.",
    "Vaccines Contain Liquid Tracking Devices - Health officials admit coronavirus vaccines contain microscopic surveillance nanobots.",
    "Cancer Cure Discovered But Big Pharma Hiding It - Natural compound eliminates all cancer types but pharmaceutical companies suppress the research.",
    "Aliens Landed in Central Park Last Night - Multiple UFOs spotted hovering over New York City before disappearing.",
    "Scientists Discover Humans Can Live 500 Years - New study proves immortality pill already perfected in underground laboratories.",
    "Lost Atlantis Found Under Arctic Ice - Researchers uncover advanced ancient civilization 5 miles below frozen ocean.",
    "Weather Control Satellites Causing All Hurricanes - Government weaponizes atmosphere to destroy specific cities.",
    "Reptilians Shapeshifting as World Leaders - Video proof shows politicians transforming into alien reptiles.",
    "Chemtrails Sterilizing Population - Airplane contrails contain chemicals designed to reduce human birth rates.",
]

print("=== TESTING FAKE NEWS EXAMPLES WITH BERT MODEL ===\n")
misclassified = 0
for i, example in enumerate(fake_news_examples, 1):
    try:
        label, confidence = predict_news(example)
        
        # Check if the prediction is correct (all examples are FAKE)
        if label == "REAL":
            misclassified += 1
            result_label = "REAL"
            print(f"{i}. {result_label} (confidence: {confidence:.1%})")
            print(f"   {example[:80]}...")
            print()
        else:
            result_label = "FAKE"
            # Optional: uncomment the line below to see successful classifications
            # print(f"{i}. {result_label} (confidence: {confidence:.1%}) - {example[:80]}...")

    except Exception as e:
        print(f"An error occurred while testing example {i}: {e}")
        print("Please ensure the model is correctly located at 'fake-news-bert-base-uncased'.")
        sys.exit(1)


print(f"\n=== SUMMARY ===")
print(f"Misclassified as REAL: {misclassified}/{len(fake_news_examples)}")
print(f"Correctly Identified as FAKE: {len(fake_news_examples) - misclassified}/{len(fake_news_examples)}")
