
class HttpException extends Error {
    status: any
    constructor (message:any, status: any) {
      super(message)
      this.status = status
      this.message  = message
      Error.captureStackTrace(this, this.constructor);
    }
  }
  
export default  HttpException