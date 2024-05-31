/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import itemModel from "../../schema/item.model";
import { cloudinary } from "../../helper/upload/cloudinary";
import { AddItem, TaskItem } from "../../interfaces";
import CategoryModel from "../../schema/category.model";
import assert from "assert";
import _, { filter, find } from "underscore";
import { Scrapping } from "../../helper/scrapping/index";
import ScrapModel from "../../schema/scrapped-items.model";
import DailyItemModel from "../../schema/daily-job.model";
import categoryModel from "../../schema/category.model";
import dailyJobModel from "../../schema/daily-job.model";

const scrap = new Scrapping();



export class ItemService {
  public async addItem(data: AddItem) {
    const {
      user,
      image_id,
      item_image,
      item_desc,
      item_title,
      item_keywords,
      item_category,
    } = data;

    const findItem = await itemModel.findOne({ item_title });

    if (findItem) {
      return {
        status: 400,
        success: false,
        message: "Item already exists",
        data: null,
      };
    }

    const newItem = await itemModel.create({
      _userId: user,
      item_title,
      item_keywords,
      item_desc,
      item_category,
      image_id: image_id,
      item_image: item_image,
    });

    //create a reverse search

    if (newItem)
      return {
        status: 200,
        success: true,
        message: `${item_title} successfully created`,
        data: newItem,
      };
  }
  public async imageUpload(data) {
    const { file, uri } = data;
    const url = uri;
    if (!file && !url) {
      return {
        status: 400,
        success: false,
        message: "No file was uploaded",
        data: null,
      };
    }

    if (file?.size > 100000000 && !url) {
      return {
        status: 400,
        success: false,
        message: "Image must not exceed 100mb",
      };
    }

    const input = url || file?.path
    const uploadImage = await cloudinary.v2.uploader.upload(input);

    return {
      status: 200,
      success: true,
      image_id: uploadImage.public_id,
      item_image: uploadImage.secure_url,
    };
  }
  public async editList(data) {
    const {
      user,
      item_id,
      item_desc,
      item_title,
      item_keywords,
      item_category,
      image_id,
      item_image,
    } = data;

    console.log(data, item_id.id);

    const updateItem = await itemModel.findOne({ _id: item_id.id });

    assert(user == updateItem._userId.toString());

    if (image_id) await cloudinary.v2.uploader.destroy(updateItem.image_id);

    updateItem.item_title = item_title || updateItem.item_title;
    updateItem.item_keywords = item_keywords || updateItem.item_keywords;
    updateItem.item_desc = item_desc || updateItem.item_desc;
    updateItem.item_category = item_category || updateItem.item_category;
    updateItem.item_image = item_image || updateItem.item_image;
    updateItem.image_id = image_id || updateItem.image_id;

    updateItem.save();

    return {
      status: 200,
      success: true,
      message: `${item_title} successfully edited`,
      data: updateItem,
    };
  }
  public async itemList(data) {

    const { user, query, } = data;
    const day = parseInt(query.days)
    const findItem = await itemModel.find({ _userId: user }).select("-_userId")
    const item_per_day = findItem.map(async (item) => {


      const others = await this.getCatalogueValue(item._id)
      return { ...item.toObject(), average: others ? others.average : '', median: others ? others.median : '', low: others ? others.low : '', high: others ? others.high : '' };
    }).filter(item => item)

    const resolved = await Promise.all(item_per_day)
    if (!findItem)
      return {
        status: 400,
        success: true,
        message: "No item found for this account",
        data: resolved,
      };

    if (findItem.length > 0)
      return {
        status: 200,
        success: true,
        message: "Items Found for this account",
        data: resolved
      };
  }
  public async paginateList(data) {

    const { query, list } = data;
    const page = parseInt(query.page);
    const limit = parseInt(query.limit);

    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;

    const results: any = {};

    if (endIndex < list.length) {
      results.next = {
        page: page + 1,
        limit: limit,
      };
    }
    if (startIndex > 0) {
      results.previous = {
        page: page - 1,
        limit: limit,
      };
    }

    const List = list.sort(
      (a: any, b: any) =>
        new Date(b.stamps).valueOf() - new Date(a.stamps).valueOf()
    );


    const CheckUrl = (results) =>
      !results
        ? null
        : `https://https://hobbyist-api.herokuapp.com/get-items?page=${results.page}&limit=${results.limit}`;

    const returningData = {
      data: List.slice(startIndex, endIndex).filter(x => x),
      PreviousPage: results.previous,
      NextUrl: CheckUrl(results.next),
      PreviousUrl: CheckUrl(results.previous),
      ListLenght: List.length,
    };

    return returningData;
  }
  public async getOneItem(data) {
    const { item_id } = data;

    const fetchItem = await itemModel.findById(item_id);
    if (!fetchItem)
      return {
        status: 400,
        success: true,
        message: "Item does not exist",
        data: {},
      };
    if (fetchItem)
      return {
        status: 200,
        success: true,
        message: "Item found",
        data: fetchItem,
      };
  }
  public async removeItem(data) {
    const { item_id } = data;

    const fetchItem = await itemModel.findById(item_id);
    if (!fetchItem)
      return {
        status: 400,
        success: true,
        message: "Item does not exist",
        data: {},
      };

    if (fetchItem) {
      const deleteItem = await itemModel.findByIdAndDelete(item_id);
      if (deleteItem)
        return {
          status: 200,
          success: true,
          message: "Item deleted",
          data: fetchItem,
        };
    }
  }
  public async addCategory(data: any) {
    const addCategory: any = await CategoryModel.findOne({});
    console.log(data);

    if (!addCategory) {
      const createList = await CategoryModel.create({
        category: [data],
      });
      return {
        status: 200,
        success: true,
        message: "Data pushed to category",
        data: createList,
      };
    } else if (addCategory) {
      const checkCategory = await CategoryModel.find({ category: data });

      if (checkCategory.length > 0) {
        return {
          status: 401,
          success: false,
          message: "Category already exists",
          data: null,
        };
      }
      if (checkCategory.length == 0) {
        addCategory.category.push(data);
        addCategory.save();

        return {
          status: 200,
          success: true,
          message: "Data pushed to category",
        };
      }
    }
  }
  public async getCategory() {
    const addCategory: any = await CategoryModel.findOne({});
    const data = !addCategory ? [] : addCategory.category;

    return {
      status: 200,
      success: true,
      message: "Category Fetched",
      data,
    };
  }
  public async getSimilarItems(item: any) {
    const getItems = await ScrapModel.findOne({ _itemId: item }).select(
      "-_id -_userId -_itemId"
    );
    const getItem = await itemModel
      .findOne({ _id: item })
      .select("-_id -_userId");

    if (!getItems && getItem) {
      return {
        status: 400,
        success: false,
        message: "No item id found",
        data: null,
      };
    }

    if (getItem && getItems) {
      return {
        status: 200,
        success: true,
        message: "Item found",
        data: {
          item: getItem,
          similar_item: getItems.similar_data.slice(0, 11),
          same_data: getItems.same_data.slice(0, 11),
        },
      };
    }
  }
  public async getRelatedItems(item: any) {
    //used for ebay items in both horizontal scroll near bottom of item page

    const getItem = await itemModel
      .findOne({ _id: item })
      .select("-_id");

    if (!getItem) {
      return {
        status: 400,
        success: false,
        message: "No item id found",
        data: null,
      };
    }

    //get daily job for id

    let dailyItem: any = await DailyItemModel.find({ _scrapId: item }).sort({ createdAt: -1 }).limit(1);
    let same_data = [], related_data = [];
    let scrapeTime = null;
    if (dailyItem && dailyItem?.length > 0) {
      same_data = dailyItem[0].same_data
      related_data = dailyItem[0].similar_data
      scrapeTime = dailyItem[0]?.createdAt
    }
    else {
      const firstScrap = await ScrapModel.findOne({ _itemId: item }).select(
        "-_id -_userId -_itemId"
      );
      same_data = firstScrap.same_data
      related_data = firstScrap.similar_data
      scrapeTime = firstScrap.createdAt;
    }
    same_data.sort((b, a) => b.price - a.price)
    related_data.sort((a, b) => b.price - a.price)
    if (getItem && same_data.length != 0) {
      return {
        status: 200,
        success: true,
        message: "Item found",
        data: {
          item: getItem,
          same_data: same_data,
          similar_item: related_data,
          scrapeTime,
        },
      };
    }

    else {
      console.log(`item "${getItem.item_title}" missing scraped data!`)
      console.log(getItem)
      console.log(same_data)
      console.log(related_data)
      return {
        status: 400,
        success: false,
        message: "No item id found",
        data: {
          item: getItem,
          same_data: [],
          similar_data: [],
          similar_item: [],
        },
      };
    }

  }
  public async saveScrapItem(data) {
    const { user, item_id, item_category, item_keyword } = data;

    const startTime = Date.now();
    await scrap.saveScrapItem(item_category, item_keyword, item_id, user);
    const endTime = Date.now();
    console.log(`Time taken to save item: ${endTime - startTime}`);

    const findItem = await itemModel.findOne({ _id: item_id });
    findItem.its_scrapped = true;
    findItem.save();

    return {
      status: 200,
      success: true,
      message: "Item scrapped",
    };
  }
  public getMedianPrice(items) {
    if (items) {

      return _.sortBy(items.map((item) => parseFloat(item.price)))[Math.floor(items.length / 2)]
    }
  }
  public getAveragePrice(items) {
    if (items) return items.map((item) => parseFloat(item.price)).reduce((a, b) => a + b, 0) / items.length
  }
  public async getDailyItems(data) {
    const { itemId, type } = data;

    if (type === "items") {
      const findItems = await DailyItemModel.find({ _scrapId: itemId });
      console.log(findItems.length);
      const findItem: any = await ScrapModel.find({ _itemId: itemId })
      const average = findItem[0]?.average
      const median = findItem[0]?.median
      const first_data = {
        itemId,
        average,
        median,
        createdAt: findItem[0]?.createdAt,
        count: 0
      }

      const data = findItems
        .map((item, count) => {
          return {
            item_id: item._scrapId,
            average: item.average,
            median: item.median,
            createdAt: item.createdAt,
            count,
          };
        })

      return {
        status: 200,
        success: true,
        message: "Resource found",
        data: [first_data, ...data].sort((a, b) => new Date(b.createdAt).valueOf() - new Date(a.createdAt).valueOf())
      }

    }
    if (type === "categories") {

      const targetItem = await itemModel.findOne({ _id: itemId });


      const targetCategory = await DailyItemModel.find({
        category: targetItem.item_category,
      });

      if (targetCategory.length === 0) {
        return {
          status: 200,
          success: true,
          message: "No resource found for this item",
          data: targetCategory,
        };
      }
      if (targetCategory.length > 0) {
        const data = targetCategory
          .map((item, count) => {
            return {
              category: item.category,
              average: item.average,
              median: item.median,
              createdAt: item.createdAt,
              count,
            };
          }).sort((a, b) => new Date(b.createdAt).valueOf() - new Date(a.createdAt).valueOf());

        return {
          status: 200,
          success: true,
          message: "Resource found",
          data
        };
      } else {
        return {
          status: 200,
          success: true,
          message: "Not enough resource found for the selected date",
          data: []
        };
      }
    }
  }
  public async getCatalogueValue(id) {

    try {

      const findScraps = await DailyItemModel.find({ _scrapId: id }).sort({ createdAt: -1 }).limit(1);
      if (findScraps.length > 0) {

        const todayValue = findScraps

        if (todayValue.length > 0) {

          //if no average you can calculate it

          let average = todayValue[0]?.average
          const median = todayValue[0]?.median
          const low = todayValue[0]?.lowest_price
          const high = todayValue[0]?.highest_price

          // (not really necessary, but just in case)
          if (!todayValue[0]?.average) {
            average = todayValue[0].reduce((a, b) => a + b, 0)
          }


          return {
            average: average ? average : '',
            median: median ? median : '',
            low: low ? low : '',
            high: high ? high : '',
          }
        }
      } else {

        const findItem: any = await ScrapModel.find({ _itemId: id }).sort({ createdAt: -1 }).limit(1)

        if (findItem) {

          const todayValue = findItem

          if (todayValue.length > 0) {
            let average = todayValue[0]?.average
            const median = todayValue[0]?.median
            const low = todayValue[0]?.lowest_price
            const high = todayValue[0]?.highest_price

            // (not really necessary, but just in case)
            if (!todayValue[0]?.average) {
              average = todayValue[0].reduce((a, b) => a + b, 0)
            }

            return {
              average: average ? average : '',
              median: median ? median : '',
              low: low ? low : '',
              high: high ? high : '',
            }
          }
        }
      }

    } catch (err) {
      console.error(err);
    }
  }
}
