import os
from typing import Optional
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Updated schema with fallback to 0.00 instead of None
class SriLankaTaxIncome(BaseModel):
    employment_income: float = Field(default=0.00, description="Annual employment income in LKR")
    investment_income: float = Field(default=0.00, description="Annual investment income in LKR")
    business_income: float = Field(default=0.00, description="Annual business income in LKR")
    rental_income: float = Field(default=0.00, description="Annual rental income in LKR")
    solar_loan_payments: float = Field(default=0.00, description="Qualifying solar loan payments in LKR")
    other_qualifying_payments: float = Field(default=0.00, description="Other qualifying relief payments in LKR")

# System prompt enforcing Sri Lankan context
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Sri Lankan tax data extractor.\n"
            "Extract ONLY the following income and deduction types if available:\n"
            "- employment_income\n"
            "- investment_income\n"
            "- business_income\n"
            "- rental_income\n"
            "- solar_loan_payments\n"
            "- other_qualifying_payments\n"
            "\n🔁 If values are given monthly, multiply by 12 to convert to annual.\n"
            "💱 Convert natural language numbers like '50 thousand', '2.5 lakhs', '1.2 million' into correct numeric amounts in LKR.\n"
            "Only return values in annual Sri Lankan Rupees (LKR)."
            "\nIf a value is not provided, set it to 0.00.\n"
            "IGNORE foreign currencies or foreign-sourced income."
        ),
        ("human", "{text}"),
    ]
)


# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

# Chain model and prompt
runnable = prompt | llm.with_structured_output(schema=SriLankaTaxIncome, method="function_calling")

# Example user inputs
user_inputs = [
    # Example 1: Mixed monthly & annual + units
    "I earn 150,000 rupees per month from my job, around 2 lakhs annually from business, and rent brings me 40 thousand per month. I also paid 1.5 lakhs on a solar panel loan this year.",

    # Example 2: All in annual terms, with 'million'
    "My employment income is about 1.8 million rupees per year. I also get 0.4 million from my shop. My investments bring in another 50,000. I paid 100,000 as other qualifying payments. No rental income.",

    # Example 3: Some incomes missing
    "I make 200,000 per month from my salary. I don’t have any rental or business income. I invested in treasury bills and got around 25 thousand for the year.",

    # Example 4: Different formats + irrelevant foreign income
    "My job in Sri Lanka pays me Rs. 175,000 monthly. I also receive Rs. 30,000 per month from an apartment I rent. I had some freelance work from abroad, paid in USD – please ignore that. Paid 50k for solar loan payments. No other income.",

    # Example 5: Vague natural phrasing
    "I get two lakhs from my employment every month. Rental gives me 20k monthly. No business or investment income. I also made a solar loan payment of 75,000 and spent 10,000 on qualifying education.",
]


# Process each input
for i, input_text in enumerate(user_inputs, 1):
    result = runnable.invoke({"text": input_text})
    print(f"\n--- User Input #{i} ---\n{input_text}\n")
    print("Extracted Income & Deductions:")
    print(result)
