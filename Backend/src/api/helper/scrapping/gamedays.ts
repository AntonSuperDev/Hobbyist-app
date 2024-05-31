import axios from 'axios'
import { logger } from '../../config/logger'
class Scrapping {
    public PriceToStr(price: string) {
        if (!price) return 0
        return parseFloat(price.slice(1).replace(",", ""))
    }
    public async getScrappingData(search_word, id) {
        const url = search_word.split(" ").join("+")
        const params = {
            api_key:
                "DCXO8PT2BDINHZNQDJUMHLK9FYAKG3MDW9U4T1A4G7KNZ4IN7WNYA796GELUFA1KW9VQ7R9ZXSXN28IH",
            url: `https://gamedaysportsmemorabilia.com/search.php?search_query=${url}`,
            // Wait for there to be at least one
            // non-empty .event-tile element
            wait_for: "body",
            extract_rules: JSON.stringify({
                data: {
                    selector: '.productGrid li.product',
                    type: "list",
                    output: {
                        title: ".card-title",
                        price: ".price--withoutTax",
                        link: {
                            selector: ".card-img-container img",
                            output: "@src"
                        },
                        url: {
                            selector: ".card-title a",
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
            logger.info(`gamedays complete with ${invs.length}`)
            return invs;
        } catch (error) {
            logger.error(`gamedays ${error}`)
            return []
        }
    }
}
export default Scrapping