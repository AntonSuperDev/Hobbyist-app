import mongoose from "mongoose";
import dotenv from "dotenv";
import { logger } from '../config/logger';

dotenv.config({ path: __dirname + '/.env' });


const connectDatabase = async () => {
  try {
    let uri = process.env.DB_CONNECTION
    const testDB = process.env.TEST_DB || 'no';
    if (testDB == 'yes' && process.env.TEST_DB_CONNECTION) uri = process.env.TEST_DB_CONNECTION
    mongoose.connect(uri, {
      //   useNewUrlParser: true,
      //   useUnifiedTopology: true,
      //   useCreateIndex: true,
    });
    const connection = mongoose.connection;
    connection.once("open", (error, db) => {
      if (error) {
        logger.error(error.message);
      } else {
        logger.info("MongoDB database connection established successfully");
      }
    });
  } catch (error) {
    logger.error(error.message);
    process.exit(1);
  }
};
export default connectDatabase;
