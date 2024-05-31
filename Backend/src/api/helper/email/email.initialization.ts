import nodemailer from 'nodemailer';

const transporter = nodemailer.createTransport({
    host: 'email-smtp.us-east-1.amazonaws.com',
    port: 587,
    secure: false, // true for 465, false for other ports
    auth: {
      user: 'AKIAVGFYWAOLF7FCKOWO', // generated ethereal user
      pass: 'BHGzegGkQcR1JXVgXa9ibdI2FG7ER5obPfZvmvExRdh3' // generated ethereal password
    }
  })
export default transporter