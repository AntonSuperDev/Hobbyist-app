import Jwt  from "jsonwebtoken";
export {};

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      DB_CONNECTION: string;
      ACCESS_TOKEN_SECRET: Jwt.Secret;
      ACCESS_TOKEN_LIFE: string;

    }
  }
}
