// /* eslint-disable @typescript-eslint/no-empty-function */
import puppeteer from "puppeteer";
import Model from "../../schema/scrapped-items.model";
import _, { sortBy } from "underscore";
import dailyJob from "../../schema/daily-job.model";
import axios, { AxiosResponse } from "axios";
import CategoryModel from "../../schema/category.model";
import itemModel from "../../schema/item.model";
import { similarity } from "../utils/similarity";
import dailyJobModel from "../../schema/daily-job.model";
import categoryModel from "../../schema/category.model";
import { logger } from '../../config/logger';
import EBay from './ebay'
import Scarce from './scarce'
import Brickowl from './brickowl'
import Chowrentoys from './chowrentoys'
import Mercari from './mercari'
import Novelship from './novelship'
import Vintage from './vintagevtg'
import Adobebooks from './adebooks'
import Bonanza from './bonanza'
import Estay from './estay'
import Gamedays from './gamedays'
import Myslabs from './myslabs'
import Steinersports from './steinersports'
import Whatnot from './whatnot'
import Yamestore from './yamestore'
import Fatherson from './fatherson'
import Amazon from './amazon'
import Bricklink from './bricklink'
import Walmat from './walmat'

class Scrapping {
  bricklink = new Bricklink()
  walmat = new Walmat()
  amazon = new Amazon()
  ebay = new EBay()
  scarce = new Scarce()
  brickowl = new Brickowl()
  chowrentoys = new Chowrentoys()
  mercari = new Mercari()
  novelship = new Novelship()
  vintage = new Vintage()
  adobebooks = new Adobebooks()
  bonanza = new Bonanza()
  estay = new Estay()
  gamedays = new Gamedays()
  myslabs = new Myslabs()
  steiner = new Steinersports()
  whatnot = new Whatnot()
  yamestore = new Yamestore()
  fatherson = new Fatherson()
  /*
  public async scrapCall(item) {
    try {
      const url = `https://hobbyist-scrapper.herokuapp.com/api/v1/scrap-item?item=${item}`;
      const response = await axios.get(url);
      return response.data;
    } catch (error) {
      console.log(error);
    }
  }
  public async ebayScrapping(item) {
    try {
      const browser = await puppeteer.launch({ args: ["--no-sandbox"] });
      const page = await browser.newPage();
      await page.goto("https://www.ebay.com/");
      await page.waitForSelector("#gh-ac");
      await page.type("#gh-ac", `${item}`);
      await page.click('input[value="Search"]');

      await page.waitForSelector("div.s-item__wrapper");

      const link = await page.$$eval("img.s-item__image-img", (items) => {
        return items.map((item: any) => {
          return item.src;
        });
      });

      const title = await page.$$eval("h3.s-item__title", (items) => {
        return items.map((item: any) => {
          return item.innerText;
        });
      });

      const price = await page.$$eval("span.s-item__price", (items) => {
        return items.map((item: any) => {
          return item.innerText;
        });
      });

      const invs = [];

      for (let i = 0, length = 17; i < length; i++) {
        const inv: any = {
          price: this.priceToStr(price[i]),
          title: title[i],
        };
        if (i < link.length) {
          inv.link = link[i];
          inv.baseCurrency = "$";
          inv.date = new Date();
        }
        invs.push(inv);
      }

      return invs;
    } catch (error) {
      if (error instanceof puppeteer.errors.TimeoutError) {
        return error.message;
      }
    }
  }
  
  public async ebayScrappingDaily(
    itemId?: any,
    user?: any,
    id?: any,
    item?: any
  ) {
    try {

      const browser = await puppeteer.launch({ args: ["--no-sandbox"] });
      const page = await browser.newPage();
      await page.goto("https://www.ebay.com/");
      await page.waitForSelector("#gh-ac");
      await page.type("#gh-ac", `${item}`);
      await page.click('input[value="Search"]');

      await page.waitForSelector("div.s-item__wrapper");

      const link = await page.$$eval("img.s-item__image-img", (items) => {
        return items.map((item: any) => {
          return item.src;
        });
      });

      const title = await page.$$eval("h3.s-item__title", (items) => {
        return items.map((item: any) => {
          return item.innerText;
        });
      });

      const price = await page.$$eval("span.s-item__price", (items) => {
        return items.map((item: any) => {
          return item.innerText;
        });
      });

      const invs = [];

      for (let i = 0, length = 17; i < length; i++) {
        const inv: any = {
          price: this.priceToStr(price[i]),
          title: title[i],
        };
        if (i < link.length) {
          inv.link = link[i];
          inv.baseCurrency = "$";
          inv.date = new Date();
        }
        invs.push(inv);
      }

      return invs;
    } catch (error) {
      console.log(error);
      throw new Error(error.message);
    }
  }
  */
  public async saveScrapItem(category, item_title, itemId, user) {

    const scrapped_data = await this.scrappingBee(item_title, itemId);
    // same_data.forEach(x => x.similarity = similarity(x.title, item_title))

    let good_items = []
    //add all similarity > 0.8, then keep on adding until count is greater than 30
    for (let i = 0; i < scrapped_data.length; i++) {
      if (scrapped_data[i].similarity >= 0.89) good_items.push(scrapped_data[i])
    }
    if (good_items.length == 0) {
      good_items = scrapped_data.slice(0, 30)
    }
    const similar_data = scrapped_data.slice(good_items.length, scrapped_data.length)

    const average = await this.getAveragePrice(good_items);
    const median = await this.getMedianPrice(good_items);
    const { high, low } = await this.getHighLowPrice(good_items)

    const createNewScrapItem = await Model.create({
      _itemId: itemId,
      _userId: user,
      similar_data: similar_data,
      same_data: good_items,
      average,
      median,
      highest_price: high,
      lowest_price: low,
    });

    if (createNewScrapItem) return true;
  }
  public async saveDailyJobforCategory() {

    // const data = await Model.find({});

    // const similar_data = data.map((item) => item.similar_data).flat()
    const categories = await CategoryModel.find({});

    const all_categories = categories[0].category


    const items = [];

    // seriously needs to be refactored
    for (let i = 0; i < all_categories.length; i += 1) {
      items.push(all_categories.slice(i, i + 1));
    }
    let response;
    let offset = 0;

    _(items).each((item) => {
      setTimeout(() => {
        item.forEach(async (category) => {
          if (category) {
            // console.log(item.title)
            const data = await this.scrappingBee(category);

            // data.forEach(el => el.similarity = similarity(el.title,item.title))

            const average = await this.getAveragePrice(data);

            const median = await this.getMedianPrice(data);

            return await dailyJob.create({
              median: median,
              average: average,
              category: category,
              similar_data: data,
            });
          }
          return response;
        });
      }, 25000 + offset);
      offset += 25000;
    });
  }
  public async saveDailyJobForItem() {

    const items_data = await itemModel.find({});
    const same_data = items_data.map((item) => {
      let item_keywords = item.item_keywords.filter(x => x.toLowerCase() != item.item_title.toLowerCase())
      let title = item.item_title
      item_keywords.forEach(x => title += " " + x)
      return {
        title: title,
        item: item._id,
      }
    })


    const items = [];

    for (let i = 0; i < same_data.length; i += 1) {
      items.push(same_data.slice(i, i + 1));
    }
    let response;
    let offset = 0;

    _(items).each((item) => {
      setTimeout(() => {
        item.forEach(async (item) => {

          if (item.title !== 'Shop on eBay') {
            // console.log(item.title)
            const data = await this.scrappingBee(item.title, item.item);

            // data.forEach(el => el.similarity = similarity(el.title, item.title))

            let whole_data = data
            let good_items = []
            //add all similarity > 0.8, then keep on adding until count is greater than 30

            for (let i = 0; i < data.length; i++) {
              if (data[i].similarity >= 0.89) good_items.push(data[i])
            }
            if (good_items.length == 0) {
              good_items = whole_data.slice(0, 30)
            }
            const similar_data = whole_data.slice(good_items.length)
            const average = await this.getAveragePrice(good_items);

            const median = await this.getMedianPrice(good_items);

            const { high, low } = await this.getHighLowPrice(good_items)


            return await dailyJob.create({
              _scrapId: item.item,
              median: median,
              average: average,
              same_data: good_items,
              similar_data: similar_data,
              lowest_price: low,
              highest_price: high,
            });
          }
          return response;
        });
      }, 25000 + offset);
      offset += 25000;
    });
  }
  public async cleanDailyJobsCategories() {
    try {
      let categories = await categoryModel.find({});
      categories = categories[0].category
      // For each item, find all daily-jobs that have the item's id
      for (const category of categories) {
        let dailyJobs = await dailyJobModel.find({ category: category })?.sort({ createdAt: -1 })

        //   Delete the title and link of all but the most recent daily-job
        dailyJobs = dailyJobs.slice(0, 4)
        for (let i = 2; i < dailyJobs.length; i++) {
          if (!(dailyJobs[i].same_data && dailyJobs[i].similar_data)) {
            continue
          }
          await dailyJobModel.updateOne({ _id: dailyJobs[i]._id }, { $unset: { same_data: '', similar_data: '' } });

        }

      }
    } catch (err) {
      logger.error(err.message);
    }
  };

  public async cleanDailyJobsItems() {

    try {

      // Find all items that belong to the user
      // const items = await db.collection('items').find({ _userId: user._id }).toArray();
      const items = await itemModel.find({})
      // For each item, find all daily-jobs that have the item's id
      for (const item of items) {
        let dailyJobs = await dailyJobModel.find({ _scrapId: item._id })?.sort({ createdAt: -1 });

        //   Delete the title and link of all but the most recent daily-job
        dailyJobs = dailyJobs.slice(0, 4)
        for (let i = 2; i < dailyJobs.length; i++) {
          if (!(dailyJobs[i].same_data && dailyJobs[i].similar_data)) {
            continue
          }
          await dailyJobModel.updateOne({ _id: dailyJobs[i]._id }, { $unset: { same_data: '', similar_data: '' } });

        }
      }
    } catch (err) {
      logger.error(err.message);
    }
  };

  public getMedianPrice(items) {
    if (items) {

      return _.sortBy(items.map((item) => parseFloat(item.price)))[Math.floor(items.length / 2)]
    }
  }
  public getAveragePrice(items) {
    if (items) return items.map((item) => parseFloat(item.price)).reduce((a, b) => a + b, 0) / items.length
  }

  public async getHighLowPrice(items) {
    items.sort((a, b) => b.price - a.price)
    const high = items[0].price
    const low = items[items.length - 1].price
    return { low, high }
  }

  public async scrappingBee(item, id = null) {
    let result: any[] = []
    result.push(...(await this.ebay.getScrappingData(item, id)))
    result.push(...(await this.scarce.getScrappingData(item, id)))
    result.push(...(await this.brickowl.getScrappingData(item, id)))
    result.push(...(await this.chowrentoys.getScrappingData(item, id)))
    result.push(...(await this.mercari.getScrappingData(item, id)))
    result.push(...(await this.novelship.getScrappingData(item, id)))
    result.push(...(await this.vintage.getScrappingData(item, id)))
    result.push(...(await this.yamestore.getScrappingData(item, id)))
    result.push(...(await this.adobebooks.getScrappingData(item, id)))
    result.push(...(await this.bonanza.getScrappingData(item, id)))
    result.push(...(await this.estay.getScrappingData(item, id)))
    result.push(...(await this.myslabs.getScrappingData(item, id)))
    result.push(...(await this.whatnot.getScrappingData(item, id)))
    result.push(...(await this.steiner.getScrappingData(item, id)))
    result.push(...(await this.gamedays.getScrappingData(item, id)))
    result.push(...(await this.fatherson.getScrappingData(item, id)))
    result.push(...(await this.amazon.getScrappingData(item, id)))
    result.push(...(await this.bricklink.getScrappingData(item, id)))
    result.push(...(await this.walmat.getScrappingData(item, id)))
    const sim_data = await axios.post('http://localhost:8000/calculate-similarity', {
      search_word: item,
      products: result.map(val => val.title)
    })
    result.forEach((element, index) => {
      element.similarity = sim_data.data.data[index]
    });
    result.sort((a, b) => b.similarity - a.similarity)
    return result.slice(0, 200)
  }
  /*
    public async testing(id) {
      console.log('testing...')
      let items_data = await itemModel.find({});
      console.log('asdfasdf')
      console.log(items_data.length)
      items_data = items_data.filter(x => x._id == id)
      if (items_data.length > 1 || items_data.length == 0) {
        console.log('fail asdfss')
        console.log(items_data.length)
        return
      }
      console.log('whhhhaaat')
      // console.log(items_data)
      // console.log(items_data[0])
      const same_data = items_data.map((item) => {
        let item_keywords = item.item_keywords.filter(x => x.toLowerCase() != item.item_title.toLocaleLowerCase())
        let title = item.item_title
        item_keywords.forEach(x => title += " " + x)
        return {
          title: title,
          item: item._id,
        }
      })
  
  
      const items = [];
  
      for (let i = 0; i < same_data.length; i += 1) {
        items.push(same_data.slice(i, i + 1));
      }
      let response;
      let offset = 0;
  
      _(items).each((item) => {
        // setTimeout(() => {
        item.forEach(async (item) => {
  
          if (item.title !== 'Shop on eBay') {
            // console.log(item.title)
            const data = await this.scrappingBeeDaily(item.title, item.item);
  
            // console.log(data)
  
            data.forEach(el => el.similarity = similarity(el.title, item.title))
  
            const average = await this.getAveragePrice(data);
  
            const median = await this.getMedianPrice(data);
  
            const { high, low } = await this.getHighLowPrice(data)
  
            console.log('creating')
            return await dailyJob.create({
              _scrapId: item.item,
              median: median,
              average: average,
              same_data: data,
              lowest_price: low,
              highest_price: high,
            });
          }
          return response;
        });
      });
      // offset += 25000;
      // });
    }
  */
}
export { Scrapping };
