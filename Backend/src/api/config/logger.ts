import winston, { format } from "winston";
const { combine, timestamp, label, printf, colorize } = format;

const appFormat = printf(({ level, message ,timestamp }) => {
  return `${timestamp} ${level}: ${message}`;
});

export const logger = winston.createLogger({
  level: "info",
  format: combine(colorize(), label({ label: "hobbyist-API" }), timestamp({format:"YYYY-MM-DD HH:mm:ss"}), appFormat),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: "app-logs.log" }),
  ],
});


