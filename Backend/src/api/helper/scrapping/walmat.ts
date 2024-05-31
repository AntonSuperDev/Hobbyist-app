import axios from 'axios'
import { logger } from '../../config/logger'
class Scrapping {
    public PriceToStr(price: string) {
        if (!price) return 0
        return parseFloat(price.split('$')[1].replace(",", ""))
    }
    public async getScrappingData(search_word, id) {
        const url = search_word.split(" ").join("+")
        const params = {
            api_key:
                "DCXO8PT2BDINHZNQDJUMHLK9FYAKG3MDW9U4T1A4G7KNZ4IN7WNYA796GELUFA1KW9VQ7R9ZXSXN28IH",
            url: `https://www.walmart.com/search?q=${url}`,
            // Wait for there to be at least one
            // non-empty .event-tile element
            wait_for: ".main-content",
            extract_rules: JSON.stringify({
                data: {
                    selector: 'div[data-testid="item-stack"]>div',
                    type: "list",
                    output: {
                        title: 'span[data-automation-id="product-title"]',
                        price: 'div[data-automation-id="product-price"]>span',
                        link: {
                            selector: "img",
                            output: "@src"
                        },
                        url: {
                            selector: "a[link-identifier]",
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
            // if(typeof response == 'object') console.log(response)
            // else console.log(response.slice(5))

            const invs: any[] = [];

            await response.map(async (item) => {
                if (item) {
                    const inv: any = {
                        price: this.PriceToStr(item.price),
                        title: item.title,
                    };
                    if (item.link) {
                        inv.link = item.link
                        inv.baseCurrency = "$";
                        inv.date = new Date();
                        inv.url = item.url
                        if (id == null) inv.category = search_word
                        inv.item = id
                    }
                    invs.push(inv);
                }
            });
            logger.info(`walmat complete with ${invs.length}`)
            return invs;
        } catch (error) {
            logger.error(`walmat ${error}`)
            return []
        }
    }
}
export default Scrapping