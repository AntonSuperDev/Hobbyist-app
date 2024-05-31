/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-empty-function */
import userModel from "../../schema/user.model";
import dailyjob from "../../schema/daily-job.model"
import item from "../../schema/item.model"
import Similaritem from "../../schema/scrapped-items.model"

import { Register, Login } from "../../interfaces";
import bcrypt from "bcryptjs";
import JWT from "jsonwebtoken";
import mailService from "../../helper/email/email.service";
import itemModel from "../../schema/item.model";
import { response } from "express";


const { ACCESS_TOKEN_SECRET } = process.env

class AuthService {
  public async Register(data: Register) {

    const { email, password, fullname, c_password } = data;

    const user = await userModel.findOne({ email: email.toLowerCase() });
    if (user) {
      throw new Error("User already exists");
    }

    if (password !== c_password) {
      throw new Error("Password does not match");
    }

    const HashPassword = await this.HashPassword(password);
    const newUser = await userModel.create({
      email: email.toLowerCase(),
      fullname,
      password: HashPassword,
    });
    const saveNewUser = await newUser.save();

    if (saveNewUser)
      return {
        success: true,
        message: "Account created successfully!",
        data,
      };
  }
  public async login(data: Login) {
    const { email, password } = data;

    const user = await userModel.findOne({ email: email.toLowerCase() });

    if (!user) {
      return {
        success: false,
        message: "Account does not exist",
        data: null,
      };
    }
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return {
        success: false,
        message: "Invalid password",
        data: null,
      };
    }
    const token = await this.accessTokenGenerator(user._id);

    return {
      success: true,
      message: "Login successful",
      data: {
        token,
        email: user.email,
        fullname: user.fullname,
        role: user.role,
      },
    };
  }
  public async HashPassword(password: any) {
    const salt = await bcrypt.genSalt(10);

    const hashedPassword = await bcrypt.hash(password, salt);

    return hashedPassword;
  }
  public async accessTokenGenerator(userId: string) {
    return JWT.sign({ _id: userId }, "swsh23hjddnknoh788778aCHOssc", {
      algorithm: "HS256",
      expiresIn: process.env.ACCESS_TOKEN_LIFE,
    });
  }
  public async sendResetCode(email: string) {
    const findUser = await userModel.findOne({ email: email.toLowerCase() });

    if (!findUser) {
      return {
        success: false,
        message: "Account does not exist",
        data: null,
      };
    }
    const resetCode = Math.floor(Math.random() * 100000);

    findUser.reset_code = resetCode;
    await findUser.save();
    //TODO: send mail_verified

    await new mailService().sendMail({
      template: {
        name: "reset.account.html",
        data: {
          Full_Name: findUser.fullname,
          resetToken: resetCode,
          userEmail: findUser.email,
        },
      },
      email: email,
      title: "Hobbyist Reset Password",
    });

    return {
      success: true,
      message: `Reset code sent ${email}`,
    };
  }
  public async resetAccount(data) {

    const { password, reset_code, c_password } = data;
    const findUser = await userModel.findOne({ reset_code });
    if (!findUser) {
      return {
        success: false,
        message: "It appears that the reset code is invalid, please try again",
        data: null,
      };
    }
    if (password !== c_password) {
      return {
        success: false,
        message: "Password does not match",
        data: null,
      };
    }

    const HashPassword = await this.HashPassword(password);
    findUser.password = HashPassword;
    findUser.reset_code = null;
    const saveResetCode = await findUser.save();
    if (saveResetCode)
      return {
        success: true,
        message: "Password reset successful",
      };
  }
  public async setRole(role: string, user) {


    const setRole = await userModel.findOne({ _id: user })
    setRole.role = role
    setRole.role_date = new Date()
    setRole.save();

    return {
      status: 200,
      success: true,
      message: `${role} set for user`
    }


  }
  public async updateUser(data) {
    const { email, fullname, user } = data;

    const find_user = await userModel.findOne({ _id: user })

    find_user.email = email
    find_user.fullname = fullname
    await find_user.save()

    return {
      status: 200,
      message: "Account updated successfully"
    }

  }
  public async get_user(data) {
    const { user } = data;

    const find_user = await userModel.findOne({ _id: user })

    return {
      status: 200,
      message: "Account updated successfully",
      data: find_user
    }


  }
  public async delete_user(data) {
    const { user } = data
    const find_delete_user = await userModel.findByIdAndDelete({ _id: user })
    const find_daily_items = await dailyjob.deleteMany({ _userId: user })
    const find_item = await item.deleteMany({ _userId: user })
    const similar = await Similaritem.deleteMany({ _userId: user })

    if (find_delete_user && find_daily_items && find_item && similar) {
      return { status: 200, success: true, message: "user deleted successfully" }
    } else {
      return { status: 400, success: false, message: 'something went wrong deleting user' }
    }

  }
}
export default AuthService;
