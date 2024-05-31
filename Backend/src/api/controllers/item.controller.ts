/* eslint-disable @typescript-eslint/no-unused-vars */
import { ItemService } from "../services/auth/items.service";
import express from "express";
import { logger } from "../config/logger";
import { AddItem } from "../interfaces";

const service = new ItemService();

const addItem = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const data: AddItem = {
      user: req?.userId,
      ...req.body,
    };
    console.log(data);
    const addItem = await service.addItem(data);
    res.status(addItem.status).send(addItem);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};
const scrapItem = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const data: AddItem = {
      user: req?.userId,
      ...req.body,
    };
    console.log(data);
    const scrapItem = await service.saveScrapItem(data);
    res.status(scrapItem.status).send(scrapItem);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};
const uploadImage = async (req: any, res: express.Response) => {
  try {
    const data = {
      file: req.file,
      uri: req.body?.uri,
    };
    const itemList = await service.imageUpload(data);
    res.status(itemList.status).send(itemList);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(500).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};
const itemList = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const data = {
      user: req?.userId,
      query: req.query,

    };
    const itemList = await service.itemList(data);
    res.status(201).send(itemList);
  } catch (error) {
    // //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};

const oneItem = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const oneItem = await service.getOneItem({ item_id: req.params.id });
    res.status(201).send(oneItem);
  } catch (error) {
    // //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};

const removeItem = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const removeItem = await service.removeItem({ item_id: req.params.id });
    res.status(201).send(removeItem);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};

const editItem = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {

    const data: AddItem = {
      user: req?.userId,
      ...req.body,
      item_id: req.params,
    };

    const editItem = await service.editList(data);
    res.status(201).send(editItem);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};

const addCategory = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {

    const { data } = req.body
    const editItem = await service.addCategory(data.toLowerCase());
    console.log(editItem);
    res.status(editItem.status).send(editItem);

  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};
const getCategory = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {

    const getItem = await service.getCategory();
    res.status(getItem.status).send(getItem);

  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
};

const getSimilarItems = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {

    //gets ebay items from daily jobs
    const getItem = await service.getRelatedItems(req.params.id);
    res.status(getItem.status).send(getItem);

  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }

}
const getDailyItems = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  const data = {
    user: req?.userId,
    itemId: req.params.itemId,
    type: req.query.type
  }
  try {

    const getItem = await service.getDailyItems(data);
    res.status(getItem.status).send(getItem);
  } catch (error) {
    //logger.error(error.message);
    console.error(error);

    res.status(400).send({
      success: false,
      message: "Something went wrong",
      reason: error.message,
      data: {},
    });
  }
}
export { addItem, itemList, oneItem, getSimilarItems, getCategory, scrapItem, removeItem, editItem, uploadImage, addCategory, getDailyItems };
