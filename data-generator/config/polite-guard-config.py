labels = ["polite", "somewhat polite", "neutral", "impolite"]

label_descriptions = """
            - polite: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.
            - somewhat polite: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.
            - neutral: Text is straightforward and factual, without emotional undertones or specific attempts at politeness.
            - impolite: Text is disrespectful or rude, often blunt or dismissive, showing a lack of consideration for the recipient's feelings."""

categories_types = {
    "travel": [
        "hotel",
        "business",
        "luxury",
        "budget or economy",
        "cultural",
        "medical",
        "air",
        "train",
        "cruises and ferries",
        "bus or car rental",
    ],
    "food and drink": [
        "pizza",
        "international",
        "regional",
        "fusion",
        "dessert",
        "vegetarian",
        "halal",
        "bakery",
        "street food",
        "buffet",
        "fast food",
        "local and organic",
        "cafe",
        "bar",
        "gluten-free",
    ],
    "stores": [
        "apparel and accessories",
        "electronics and appliances",
        "grocery and food",
        "health and beauty",
        "home and furniture",
        "sports and outdoors",
        "toys and games",
    ],
    "finance": ["banking", "credit", "insurance", "loans", "fees and charges"],
    "professional development": [
        "technical skills",
        "soft skills",
        "creative skills",
        "workshop",
        "bootcamp",
        "integration training",
    ],
    "sports clubs": [
        "team sports",
        "individual sports",
        "racket sports",
        "water sports",
        "winter sports",
        "combat sports",
    ],
    "cultural and educational": [
        "museum",
        "theater",
        "zoo or aquarium",
        "art gallery",
        "botanical garden",
        "library",
    ],
}

use_case = "customer service chatbots"

prompt_examples = """ 
            LABEL: polite
            CATEGORY: food and drink
            TYPE: cafe
            OUTPUT: Thank you for visiting! While we prepare your coffee, feel free to relax or browse our selection of pastries. Let us know if we can make your day even better!
            REASONING: This text is polite because it expresses gratitude and encourages the customer to feel at ease with a welcoming tone. 
            Phrases like "Let us know if we can make your day even better" show warmth and consideration, enhancing the customer experience.
            
            LABEL: somewhat polite
            CATEGORY: travel
            TYPE: train
            OUTPUT: I understand your concern about your booking, and I'll check what options we have for you.
            REASONING: This text would be classified as "somewhat polite."
            The acknowledgment of the customer's concern shows a basic level of respect.
            The sentence is direct and lacks additional warmth or formality, but it communicates a willingness to help.
            The use of "I'll check" is a straightforward commitment to action without additional courteous phrases that would make it fully polite.
            
            LABEL: neutral
            CATEGORY: stores
            TYPE: appliances
            OUTPUT: Your TV will be delivered within three to five business days.
            REASONING: This text would be classified as "neutral."
            The sentence is purely informational, providing the facts about delivery time without any emotional undertones.
            There are no phrases that express politeness or rudeness; it's a straightforward statement.
            The tone is impersonal and focused solely on conveying the necessary information.

            LABEL: impolite
            CATEGORY: sports clubs
            TYPE: team sports
            OUTPUT: We might as well be playing with one less person. At least then we wouldn't be expecting anything from you.
            REASONING: This statement is impolite because it outright dismisses the person's usefulness. It's demeaning and emphasizes that their presence adds 
            no value, making it deeply discouraging.
            """
