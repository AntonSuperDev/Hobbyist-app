import mongoose from "mongoose";

const Schema = new mongoose.Schema(
  {
    _userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      ref: "Users",
    },
    item_title: {
      type: String,
      required: true,
    },
    item_keywords: {
      type: Array,
      required: true,
    },
    item_desc: {
      type: String,
    },
    item_category: {
      type: String,
      required: true,
    },
    its_scrapped: {
      type: Boolean,
    },
    item_image: {
      type: String,
    },
    image_id: {
      type: String,
    },
    stamps: {
      type: Date,
      default: Date.now,
    }
  },
  { timestamps: true }
);

Schema.index({ _userId: 1, type: -1 });

export default mongoose.model("item", Schema);
