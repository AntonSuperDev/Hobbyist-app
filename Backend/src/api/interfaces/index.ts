export interface Register{
    email: string;
    password: string;
    fullname: string;
    c_password: string;
}

export interface Login {

    email: string;
    password: string;
}

export interface AddItem {
    user: string;
    file:any,
    item_desc: string;
    item_category: string;
    item_title: string;
    item_keywords:Array<string>
    image_id:string, 
    item_image: string
}

export class TaskItem {
    
    public stamps: Date;
}