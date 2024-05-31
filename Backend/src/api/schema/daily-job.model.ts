import mongoose from "mongoose";

const Schema = new mongoose.Schema(
  {
    _itemId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Items",
    },
    _userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Users",
    },
    _scrapId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Similar-item",
    },
    median: {
      type: Number,
    },
    average: {
      type: Number,
    },
    highest_price: {
      type: Number,
    },
    lowest_price: {
      type: Number,
    },
    category: {
      type: String,
    },
    same_data: [],
    similar_data: [],
  },
  { timestamps: true }
);

Schema.index({ _itemId: 1, type: -1 });

export default mongoose.model("daily-job", Schema);
