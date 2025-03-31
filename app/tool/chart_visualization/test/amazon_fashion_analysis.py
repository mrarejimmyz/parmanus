import asyncio

from app.agent.data_analysis import DataAnalysis

# from app.agent.manus import Manus


async def main():
    agent = DataAnalysis()
    # agent = Manus()
    await agent.run(
        """Here's last month's sales data from my Amazon store in './data/amazon_sales_jan2025.xlsx'. Could you analyze it？  """
    )


if __name__ == "__main__":
    asyncio.run(main())
