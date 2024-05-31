/* eslint-disable @typescript-eslint/no-var-requires */
import { logger } from "../../config/logger"
import { Scrapping } from "../scrapping/index"
// import connectDatabase from '../../database';

function removePunctuation(s){
    return s.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").replace(/\s{2,}/g," ")
}

const similarity = (matching_title,target_title)=>{
    try{
        const natural = require('natural');
        const sw = require('remove-stopwords')

        matching_title = removePunctuation(matching_title)
        target_title = removePunctuation(target_title)

        target_title = target_title.split(' ')
        target_title = sw.removeStopwords(target_title)

        matching_title = matching_title.split(' ')
        matching_title = sw.removeStopwords(matching_title)

        matching_title = matching_title.map(word=>natural.PorterStemmer.stem(word))
        target_title = target_title.map(word=>natural.PorterStemmer.stem(word))

        const intersection: number = matching_title.filter(word => target_title.includes(word)).length

        if(matching_title.length == 0) return 0
        return intersection/(matching_title.length + target_title.length - intersection)
    }
    catch(e){
        console.log("error in calculating similarity")
        logger.error("error in calculating sim: ")
        console.log(e)
        logger.error(e)
        console.log(matching_title, target_title)
        logger.error(matching_title,target_title)
        return 0
    }

}

const test = ()=>{
    console.log('running test...')
    let t = "Ring"
    const m = "Feng Shui Pixiu Mani Mantra Adjustable Ring Protection Wealth Ring Qua…"

    console.log(similarity(m,t));
}


//test a daily scrape
// const test_scrape = ()=>{


//     // connectDatabase()
//     let id = '62ea8ecd5cf90b24dcfee9e0'
//     const scrap = new Scrapping()
//     scrap.testing(id)

// }
// test()

export {similarity}