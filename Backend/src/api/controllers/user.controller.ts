import AuthService from "../services/auth/user.service";
import express from "express";
import { logger } from "../config/logger";

const authService = new AuthService();


const register = async (
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const register = await authService.Register(req.body);
    res.status(201).send(register);
  } catch (error) {
    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });
  }
};
const login = async (
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const login = await authService.login(req.body);
    res.status(201).send(login);
  } catch (error) {
    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });
  }
};
const sendResetCode = async (
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const { email } = req.body;
    const sendResetCode = await authService.sendResetCode(email);
    res.status(201).send(sendResetCode);
  } catch (error) {
    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });
  }
};

const resetAccount = async (
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) => {
  try {

    const resetAccount = await authService.resetAccount(req.body);
    res.status(201).send(resetAccount);
  } catch (error) {
    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });
  }
};

const setRole = async (
  req: any,
  res: express.Response,
  next: express.NextFunction
) => {
  try {
    const { role } = req.body
    const resetAccount = await authService.setRole(role, req?.userId);
    res.status(200).send(resetAccount);
  } catch (error) {
    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });
  }
};

const updateUser = async (req: any,
  res: express.Response,
  next: express.NextFunction) => {
  try {

    const user = req.userId
    const { email, fullname } = req.body

    let data = { user, email, fullname }

    const update_user = await authService.updateUser(data)
    res.status(200).send(update_user)
  } catch (error) {

    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });

  }

}


const get_user = async (req: any,
  res: express.Response,
  next: express.NextFunction) => {
  try {
    const user = req.userId
    let data = { user }

    const get_user = await authService.get_user(data)
    res.status(get_user.status).send(get_user)
  } catch (error) {

    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });

  }
}

const delete_user = async (req: any,
  res: express.Response,
  next: express.NextFunction) => {
  try {
    const user = req.userId
    let data = { user }

    const get_user = await authService.delete_user(data)
    res.status(get_user.status).send(get_user)
  } catch (error) {

    logger.error(error.message);
    res.status(400).send({
      success: false,
      message: error.message,
      data: null,
    });

  }

}
export { register, login, sendResetCode, resetAccount, setRole, updateUser, get_user, delete_user };
