import mongoose from "mongoose";

const Schema = new mongoose.Schema(
  {
    _itemId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      ref: "Items",
    },
    _userId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        ref: "Users",
    },
    same_data:[],
    similar_data:[],
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
    }
    
  },
  { timestamps: true }
);

Schema.index({ _itemId: 1, type: -1 });

export default mongoose.model("Similar-item", Schema);
