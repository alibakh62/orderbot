# orderbot

OrderBot is a virtual salesperson that take a product inquiry from the customer and fulfills the order.

OrderBot features:
- A voice interface
- Easy adaptation to different formats of product catalogs
- Ability to handle complex customer inquiries, e.g., “I wanna have a healthy lunch. Give me a few options, please?”
- Enhanced product search by using product features
- Customizable order fulfillment process. Currently supporting two customizations:
  - Boost products on promotion
  - Cross-sell suggestions

# Installation
- Rename the `.evn.example` file to `.env` and fill in the required environment variables.
- Run `pip install -r requirements.txt` to install the required packages.

# Usage
- First, you need to create a vector representation of the products in the catalog. Run `python index.py` to create the vector DB. Make sure to have the product catalog file in the `data` folder.
  - **Note:** As of now, we only support CSV file ingestion. The CSV file should have the following columns: `index`, `PRODUCT_NAME`.
- Run `python -m streamlit run voice_bot.py` to start the bot.



