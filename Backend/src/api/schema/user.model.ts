import mongoose from "mongoose"


const Schema = new mongoose.Schema({
    fullname: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
    },
    password: {
        type: String,
        required: true,
    },
    mail_verified: {
        type: Boolean,
        default: false
    },
    role: {
        type: String,
        enum: ['tier-I', 'tier-II', 'tier-III'],
        default: 'tier-I'
    },
    role_date: {
        type: Date,
    },
    reset_code: {
        type: String,
        default: null,
        expires: "10m"
    }
}, { timestamps: true })

Schema.index({ _id: 1, type: -1 })

export default mongoose.model('User', Schema)
