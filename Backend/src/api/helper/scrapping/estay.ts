import axios from 'axios'
import { logger } from '../../config/logger';
class Scrapping {
    public PriceToStr(price: string) {
        price = price.replace(/ /g, "").replace(/,/g, "")
        let i = 0
        for (i = 0; i < price.length; i++)
            if (price[i] >= '0' && price[i] <= '9')
                break;
        if (i == price.length)
            return {
                currency: 'No',
                price: 0
            }
        return {
            currency: price.slice(0, i),
            price: parseFloat(price.slice(i))
        }
    }
    public async getScrappingData(search_word, id) {
        const url = search_word.split(" ").map(val=>encodeURIComponent(val)).join(" ")
        const params = {
            api_key:
                "DCXO8PT2BDINHZNQDJUMHLK9FYAKG3MDW9U4T1A4G7KNZ4IN7WNYA796GELUFA1KW9VQ7R9ZXSXN28IH",
            url: `https://www.etsy.com/search?q=${url}&ref=search_bar`,
            // Wait for there to be at least one
            // non-empty .event-tile element
            wait_for: "div[data-search-results-container]",
            extract_rules: JSON.stringify({
                data: {
                    selector: 'ol li',
                    type: "list",
                    output: {
                        title: 'h3.v2-listing-card__title',
                        baseCurrency: ".currency-symbol",
                        price: ".currency-value",
                        link: {
                            selector: "img[data-listing-card-listing-image]",
                            output: "@src"
                        },
                        url: {
                            selector: "a",
                            output: "@href"
                        }
                    }
                }
            }),
        }
        try {
            const { data } = await axios.get("https://app.scrapingbee.com/api/v1/", {
                params: params
            });

            const response = data.data;

            const invs: any[] = [];

            await response.map(async (item) => {
                item.date = new Date();
                if (id == null) item.category = search_word
                item.item = id
                invs.push(item);
            });
            logger.info(`estay complete with ${invs.length}`)
            return invs;
        } catch (error) {
            logger.error(`estay ${error}`)
            return []
        }
    }
}
export default Scrapping