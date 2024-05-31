import axios from 'axios'
import { logger } from '../../config/logger';
class Scrapping {
    public priceToStr(price) {
        if (price && price?.includes("to")) {
            const lomerImit = price
                .split("$").join("")
                .replace("to", "").replace(",", "")
                .split(" ")[0];
            const upperLimit = price
                .split("$").join("")
                .replace("to", "").replace(",", "")
                .split(" ")[1];
            const avgPrice = (parseFloat(lomerImit) + parseFloat(upperLimit)) / 2;
            return !avgPrice ? lomerImit : avgPrice;
        } else {
            return parseFloat(price?.replace("$", "").replace(",", ""));
        }
    }
    public async getScrappingData(search_word, id) {
        const url = search_word.split(" ").join("+")
        const params = {
            api_key:
                "DCXO8PT2BDINHZNQDJUMHLK9FYAKG3MDW9U4T1A4G7KNZ4IN7WNYA796GELUFA1KW9VQ7R9ZXSXN28IH",
            url: `https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2380057.m570.l1313&_nkw=${url}&_sacat=0`,
            // Wait for there to be at least one
            // non-empty .event-tile element
            wait_for: ".s-item",
            extract_rules: JSON.stringify({
                data: {
                    // Lets create a list with data
                    // extracted from the .event-tile element
                    selector: ".s-item",
                    type: "list",
                    // Each object in the list should
                    output: {
                        // have a title lifted from
                        // the .event-tile__title element
                        price: ".s-item__price",
                        title: ".s-item__title",
                        link: {
                            selector: ".s-item__image-wrapper img",
                            output: "@src",
                        },
                        url: {
                            selector: ".s-item__link",
                            output: "@href"
                        }

                    },
                },
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
                        price: this.priceToStr(item.price),
                        title: item.title,
                    };
                    if (item.link) {
                        inv.link = item.link;
                        inv.baseCurrency = "$";
                        inv.date = new Date();
                        inv.url = item.url
                        if (id == null) inv.category = search_word
                        inv.item = id
                    }
                    invs.push(inv);
                }
            });

            logger.info(`ebay complete with ${invs.length}`)
            return invs.slice(1);
        } catch (error) {
            logger.error(`ebay ${error}`)
            return []
        }
    }
}
export default Scrapping