import warnings

warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


health_context = """
1_Classic Potato Chips_150 calories, 10g fat, 2g protein_Potatoes, vegetable oil, salt
2_Cola Soda 500ml_140 calories, 0g fat, 0g protein_Carbonated water, high fructose corn syrup, caramel color, phosphoric acid, caffeine
3_Whole Wheat Bread_100 calories, 1g fat, 4g protein_Whole wheat flour, water, yeast, salt, sugar
4_Organic Bananas_89 calories, 0.3g fat, 1.1g protein_Bananas
5_Fresh Ground Coffee_5 calories, 0g fat, 0.3g protein_Coffee beans
"""

promotion_context = """
1_Classic Potato Chips_Buy 2 Get 1 Free
2_Cola Soda 500ml_NA
3_Whole Wheat Bread_NA
4_Organic Bananas_2 for $1
5_Fresh Ground Coffee_20% off
"""

product_context = """
1_Classic Potato Chips_Snacks_Chips
2_Cola Soda 500ml_Beverages_Soft Drinks
3_Whole Wheat Bread_Bakery_Bread
4_Organic Bananas_Produce_Fruit
5_Fresh Ground Coffee_Beverages_Coffee
"""

price_context = """
1_Classic Potato Chips_$1.99 
2_Cola Soda 500ml_$1.50 
3_Whole Wheat Bread_$2.99 
4_Organic Bananas_$0.59 
5_Fresh Ground Coffee_$4.99 
"""

intent_detection_chain = (
    PromptTemplate.from_template(
        """Your job is to identify the intent of the user. The user will be asking about a product and or a bunch of products. The user may want to know different things about the products. Given the user question below, classify the user's intent as 'HEALTH', 'PRICE', 'NUTRITION', 'INGREDIENTS', 'PROMOTIONS', 'PLACING_ORDER'.
    <question>
    {question}
    </question>
    Classification:"""
    )
    | llm
    | StrOutputParser()
)


price_chain = (
    PromptTemplate.from_template(
        """Your job is to extract the price of the product from the price context data below inside triple backticks: \
        PRICE CONTEXT
        ```
        1_Classic Potato Chips_$1.99 
        2_Cola Soda 500ml_$1.50 
        3_Whole Wheat Bread_$2.99 
        4_Organic Bananas_$0.59 
        5_Fresh Ground Coffee_$4.99 
        ```
        The product price data has three features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, and the third feature is PRODUCT PRICE. Given the customer question below, identify the price of the product. If the product name does not exist in the product price data, return 'Product not found'. DO NOT make up any price. Only respond with product price.\
        <question>
        {question}
        </question>
        Price:"""
    )
    | llm
    | StrOutputParser()
)  # Expected output: $1.99

order_chain = (
    PromptTemplate.from_template(
        """Your job is to identify the product from the product context information below. \
        PRODUCT CONTEXT
        ```
        1_Classic Potato Chips_Snacks_Chips
        2_Cola Soda 500ml_Beverages_Soft Drinks
        3_Whole Wheat Bread_Bakery_Bread
        4_Organic Bananas_Produce_Fruit
        5_Fresh Ground Coffee_Beverages_Coffee
        ```
        The product context data has four features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, the third feature is PRODUCT CATEGORY, and the fourth feature is PRODUCT SUBCATEGORY. Given the customer question below, identify the product name. If the product name does not exist in the product context data, return 'Product not found'. \
        <question>
        {question}
        </question>
        Product name:"""
    )
    | llm
    | StrOutputParser()
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:
        Question: {question}
        Answer:"""
    )
    | llm
    | StrOutputParser()
)

branch = RunnableBranch(
    (lambda x: "price" in x["intent"].lower(), price_chain),
    (lambda x: "placing_order" in x["intent"].lower(), order_chain),
    general_chain,
)

QUESTION = "What's the price of Hummus?"

print("Customer Intent: " + intent_detection_chain.invoke({"question": QUESTION}))

print("Product Price: " + price_chain.invoke({"question": QUESTION}))

full_chain = {
    "intent": intent_detection_chain,
    "question": lambda x: x["question"],
} | branch

print("Full chain: " + full_chain.invoke({"question": QUESTION}))
