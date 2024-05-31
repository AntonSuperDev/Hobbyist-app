/* eslint-disable no-useless-escape */
/* eslint-disable @typescript-eslint/no-empty-function */
import { readFileSync } from "fs";
import * as path from "path";
import HttpException from "../handler/error.handler";
import { MailTemplateDto } from "./email.dto";
import { logger } from "../../config/logger";
import transporter from "./email.initialization";

export default class Email {
  constructor() {}
  public getEmailTemplate(template: string) {
    try {
      const file = path.join(__dirname, `./templates/${template}`);
      const emailTemplate = readFileSync(file, "utf8");
      return emailTemplate;
    } catch (error) {
      logger.error(error);
      return "An error occurred while loading template";
    }
  }
  public parseEmailTemplate(emailTemplate) {
    let template = this.getEmailTemplate(emailTemplate.name);
    if (!template) throw new HttpException("email template not found", 500);

    Object.keys(emailTemplate.data).forEach((key) => {
      const regex = new RegExp(`{{\.\*${key}\.\*}}`, "g");
      template = template.replace(regex, emailTemplate.data[key]);
    });

    return template;
  }
  public async sendMail (mailData) {
    try {
      const parsedTemplate = this.parseEmailTemplate(mailData.template)

      const message = {
        from: `${mailData.title} <test.server.shadow@gmail.com>`,
        to: mailData.email,
        subject: 'Hobbyist',
        html: parsedTemplate
      }
      const info = await transporter.sendMail(message)

      return info
    } catch (error) {
      throw new HttpException(error.toString(), 500)
    }
  }
}
