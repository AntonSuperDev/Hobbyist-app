import express from "express";
const router = express.Router();

import {
  addItem,
  itemList,
  oneItem,
  removeItem,
  editItem,
  addCategory,
  uploadImage,
  getCategory,
  getSimilarItems,
  scrapItem,
  getDailyItems
} from "../../controllers/item.controller";
import {uploads}from "../../middleware/multer";
import { verify } from "../../middleware/verify.token";

router.post("/api/v1/add-item", verify, addItem);
router.put("/api/v1/edit-item/:id", verify, editItem);
router.post("/api/v1/item-image", verify, uploads.single("file"), uploadImage)
router.post("/api/v1/item-image-url", verify, uploadImage)
router.get("/api/v1/items-list", verify, itemList);
router.get("/api/v1/get-item/:id", verify, oneItem);
router.delete("/api/v1/delete-item/:id", verify, removeItem);
router.post("/api/v1/add-category", verify, addCategory);
router.get("/api/v1/get-category", verify, getCategory);
router.get("/api/v1/get-similar-item/:id", verify, getSimilarItems);
router.post("/api/v1/scrap-item", verify, scrapItem);
router.get("/api/v1/get-daily-items/:itemId", verify, getDailyItems);


// router.post("/api/v1/signin", login);
// router.post("/api/v1/send-reset-code", sendResetCode);
// router.post("/api/v1/reset-account", resetAccount);

export default router;
