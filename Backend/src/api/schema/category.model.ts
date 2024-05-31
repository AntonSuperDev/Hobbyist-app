import mongoose from "mongoose"


const Schema = new mongoose.Schema({
    category: [{
        type: String
    }]
}, { timestamps: true })

Schema.index({_id: 1, type: -1})

export default mongoose.model('Category', Schema)
