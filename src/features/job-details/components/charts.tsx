/* eslint-disable prettier/prettier */
/* eslint-disable radix */
/* eslint-disable react-native/no-inline-styles */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-unused-vars */

import {
  ChartBackground,
  ChartFilterContainer,
  ChartFilterText,
  ChartIconImage,
  ChartItemLabelText,
  ChartLegend,
  ChartLegendGroup,
  ChartLegendlabel,
  ChartLegendlabelText,
  ChartLegendlabelTextActive,
  ChartTimeLabelText,
  ChartTimeLabelTextDivider,
  OnTouch,
} from './chart.styles';
import React, {FC, useCallback, useEffect, useState} from 'react';

import AsyncStorage from '@react-native-async-storage/async-storage';
import ChartIcon from '../../../../assets/chartind.png';
import ChartIcon1 from '../../../../assets/chartind1.png';
import ChartIconBlack from '../../../../assets/chartBlack.png';
import ChartIconLightBlue from '../../../../assets/chartLightBlue.png';
import ChartIconRed from '../../../../assets/chartRed.png';
import {Dimensions} from 'react-native';
import {LineChart} from 'react-native-chart-kit';
import {Loading} from '../../auth/components/login.styles';
import Moment from 'moment';
import {Spacer} from '../../../components/utility/spacer';
import {Text} from 'react-native';
import {View} from 'react-native';
import {parse} from 'react-native-svg';
import { BASE_URL } from '../../../config';

const getWidth = Dimensions.get('window').width;
const today = new Date();
interface Props {
  onItem: any;
  onCat: any;
  item_id: any;
  user: any;
  navigation: any;
  fullitem: any;
}
// const date = new Date(today).toLocaleDateString("de");
const Chart: FC<Props> = ({onItem, onCat, item_id, user, navigation, fullitem}) => {
  const [avgOnItem, setSetAvgOnItem] = useState(true);
  const [avgOnCat, setSetAvgOnCat] = useState(true);
  const [medianOnCat, setSetMedianOnCat] = useState(true);
  const [medianOnItem, setSetMedianOnItem] = useState(true);
  const [dataOnItem, setDataOnItem] = useState([]);
  const [dataOnCat, setDataOnCat] = useState([]);
  const [wrongLength, setWronglength] = useState<any | null>([]);
  const [dummy, setDummy] = useState<any | null>([]);
  // const [firstDateOnCat, setFirstDateOnCat] = React.useState('');
  const [count, setCount] = useState('10');
  const [itemLength, setItemLength] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  // const todayDate = new Date();
  // const [chart, setChart] = useState<string[], never[]>();
  // const [datasetts, setDatasetts]
  const getDaily = async () => {
    // setIsLoading(true);
    setDataOnCat([]);
    setDataOnItem([]);
    setWronglength([]);
    // setItemLength()
    const token = await AsyncStorage.getItem('token');
    const id = item_id;
    var itemApiUrl = `${BASE_URL}/api/v1/get-daily-items/${id}?type=items`;
    var catApiUrl = `${BASE_URL}/api/v1/get-daily-items/${id}?type=categories`;

    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    await fetch(catApiUrl, {
      method: 'GET',
      headers: headers,
    })
      .then(res => res.json())
      .then(res => {
        setItemLength(res.data.length);
        // console.log(res.data.length);
        // res.data.filter()
        // var filtered = res.data.filter((a: any) => {
        //   var date = Moment(Date.parse(a.createdAt));
        //   return (
        //     date >= Moment(today).subtract(count, 'day') &&
        //     date <= Moment(today)
        //   );
        // });
        // if (res.data.length > count){
        //   setDataOnCat(res.data.slice(0,count));
        // } else {
        setDataOnCat(res.data);
        // }
        // console.log(filtered.length);

        // var newItems30 = res.data.filter((a: any) => {
        //   var date = Moment(Date.parse(a.createdAt));
        //   return (
        //     date >= Moment(today).subtract(30, 'day') &&
        //     date <= Moment(today)
        //   );
        // });
        // setDataOnCat30(newItems30);

        // var newItems180 = res.data.filter((a: any) => {
        //   var date = Moment(Date.parse(a.createdAt));
        //   return (
        //     date >= Moment(today).subtract(180, 'day') &&
        //     date <= Moment(today)
        //   );
        // });
        // setDataOnCat180(newItems180);

        // var newItems180 = res.data.filter((a: any) => {
        //   var date = Moment(Date.parse(a.createdAt));
        //   return (
        //     date >= Moment(today).subtract(180, 'day') &&
        //     date <= Moment(today)
        //   );
        // });
        // setDataOnCat180(newItems180);

        // var newItems360 = res.data.filter((a: any) => {
        //   var date = Moment(Date.parse(a.createdAt));
        //   return (
        //     date >= Moment(today).subtract(360, 'day') &&
        //     date <= Moment(today)
        //   );
        // });
        // setDataOnCat360(newItems360);
      })
      .catch(err => {
        console.log(err);
        // setIsLoading(false);
      });
    fetch(itemApiUrl, {
      method: 'GET',
      headers: headers,
    })
      .then(response => response.json())
      .then(response => {
        if (response.data.length < parseInt(count)) {
          setWronglength([]);
          setDummy([]);
          console.log(parseInt(count) - response.data.length);
          for (
            let index = 0;
            index < parseInt(count) - response.data.length;
            index++
          ) {
            // setWronglength([]);
            dummy.push(0);
          }
          setDataOnItem(response.data);
          setWronglength(dummy);
          // dataOnItem.forEach(element => {

          // });
        } else {
          setDataOnItem(response.data.slice(0, parseInt(count)).reverse());
          // setWronglength(wrongLength);
        }

        // console.log(response.data);
        setIsLoading(false);
      })
      .catch(err => {
        console.log(err);
        // setIsLoading(false);
      });
    setIsLoading(false);
  };

  useEffect(() => {
    getDaily();
  }, []);

  const ChartData = {
    labels: [
      `${Moment(today).subtract(count, 'day').format('MM/DD/YYYY')}`,
      ...dataOnCat.slice(0, parseInt(count) - 2).map((item: any) => ''),
      `${Moment(today).format('MM/DD/YYYY')}`,
    ],
    // labels: dataOnItem.map((item: any) => item.median),
    // dataOnItem.length < parseInt(count) && dataOnCat.slice(0, (parseInt(count) - dataOnItem.length)).map((item: any) => 0)
    datasets: [
      medianOnItem && dataOnItem.length !== 0
        ? {
            data: dataOnItem.length
              ? [
                  ...wrongLength.slice(0, wrongLength.length),
                  ...dataOnItem.map((item: any) => item.median),
                ]
              : [0, 0],
            strokeWidth: 1,
            color: () => '#81BDCA',
          }
        : {
            data: dataOnItem.length ? [0, 0] : [0, 0],
            strokeWidth: 1,
            color: () => 'transparent',
          },
      avgOnItem && dataOnItem.length !== 0
        ? {
            data: dataOnItem.length
              ? [
                  ...wrongLength.slice(0, wrongLength.length),
                  ...dataOnItem.map((item: any) => item.average),
                ]
              : [0, 0],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => '#FD7474',
          }
        : {
            data: dataOnItem.length ? [0, 0] : [0, 0],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => 'transparent',
          },
      avgOnCat
        ? {
            data:
              user === 'tier-III'
                ? dataOnCat.length
                  ? dataOnCat
                      .slice(0, parseInt(count))
                      .map((item: any) => item.average)
                  : [0, 0]
                : [0, 0],
            // data: [220, 230],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => '#000000',
          }
        : {
            data: dataOnCat.length ? [0, 0] : [0, 0],
            // data: [220, 230],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => 'transparent',
          },
      medianOnCat
        ? {
            data:
              user === 'tier-III'
                ? dataOnCat.length
                  ? dataOnCat
                      .slice(0, parseInt(count))
                      .map((item: any) => item.median)
                  : [0, 0]
                : [0, 0],
            // data: [200, 210],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => '#1cc88a',
          }
        : {
            data: dataOnCat.length ? [0, 0] : [0, 0],
            // data: [200, 210],
            strokeWidth: 1,
            strokeDasharray: 2,
            color: () => 'transparent',
          },
    ],
  };
  const chartConfig = {
    backgroundColor: '#1cc910',
    backgroundGradientFromOpacity: 0,
    backgroundGradientToOpacity: 0,
    fillShadowGradientToOpacity: 0,
    fillShadowGradientFromOffset: 0,
    fillShadowGradientFromOpacity: 0,
    useShadowColorFromDataset: false,
    color: (opacity = 0) => `rgba(0, 0, 0, ${opacity})`,
    // labelColor: (opacity = 0.7) => `rgba(255, 255, 255, 0.72)`,
    decimalPlaces: 2,
    propsForDots: {
      r: '1',
      strokeWidth: '6',
    },
    style: {
      borderRadius: 1,
      // color: 'black',
    },
  };

  const chartStyle = {
    marginVertical: 8,
    marginLeft: -30,
    // marginRight: 10,
  };

  return (
    <>
      <Spacer />
      <ChartLegend>
        <ChartLegendGroup>
          <ChartLegendlabel>
            <OnTouch onPress={() => setSetAvgOnItem(!avgOnItem)}>
              <ChartIconImage
                source={
                  avgOnItem ? JSON.parse(ChartIconRed) : JSON.parse(ChartIcon1)
                }
              />
            </OnTouch>
            {avgOnItem ? (
              <ChartLegendlabelTextActive>
                Average of Item
              </ChartLegendlabelTextActive>
            ) : (
              <ChartLegendlabelText>Average of Item</ChartLegendlabelText>
            )}
          </ChartLegendlabel>
          {user === 'tier-III' ? (
            <ChartLegendlabel>
              <OnTouch onPress={() => setSetAvgOnCat(!avgOnCat)}>
                <ChartIconImage
                  source={
                    avgOnCat
                      ? JSON.parse(ChartIconBlack)
                      : JSON.parse(ChartIcon1)
                  }
                />
              </OnTouch>
              <ChartLegendlabelText>Average of Category</ChartLegendlabelText>
            </ChartLegendlabel>
          ) : (
            <View
              style={{
                flex: 1,
                // position: 'absolute',
                left: 0,
                top: 0,
                opacity: 0.1,
                padding: 10,
                backgroundColor: 'black',
                width: getWidth,
              }}>
              <OnTouch
                onPress={() => {
                  navigation.navigate('AccountMgmt');
                }}>
                <ChartLegendlabel>
                  <OnTouch onPress={() => setSetAvgOnCat(!avgOnCat)}>
                    <ChartIconImage
                      source={
                        avgOnCat
                          ? JSON.parse(ChartIcon)
                          : JSON.parse(ChartIcon1)
                      }
                    />
                  </OnTouch>
                  <ChartLegendlabelText>
                    Average of Category
                  </ChartLegendlabelText>
                </ChartLegendlabel>
              </OnTouch>
            </View>
          )}
        </ChartLegendGroup>
        <ChartLegendGroup>
          <ChartLegendlabel>
            <OnTouch onPress={() => setSetMedianOnItem(!medianOnItem)}>
              <ChartIconImage
                source={
                  medianOnItem
                    ? JSON.parse(ChartIconLightBlue)
                    : JSON.parse(ChartIcon1)
                }
              />
            </OnTouch>
            <ChartLegendlabelText>Median of Item</ChartLegendlabelText>
          </ChartLegendlabel>
          {user === 'tier-III' ? (
            <ChartLegendlabel>
              <OnTouch onPress={() => setSetMedianOnCat(!medianOnCat)}>
                <ChartIconImage
                  source={
                    medianOnCat ? JSON.parse(ChartIcon) : JSON.parse(ChartIcon1)
                  }
                />
              </OnTouch>
              <ChartLegendlabelText>Median of Category</ChartLegendlabelText>
            </ChartLegendlabel>
          ) : (
            <View
              style={{
                flex: 1,
                // position: 'absolute',
                left: 0,
                top: 0,
                opacity: 0.1,
                padding: 10,
                backgroundColor: 'black',
                width: getWidth,
              }}>
              <OnTouch
                onPress={() => {
                  navigation.navigate('AccountMgmt');
                }}>
                <ChartLegendlabel>
                  <OnTouch onPress={() => setSetMedianOnCat(!medianOnCat)}>
                    <ChartIconImage
                      source={
                        medianOnCat
                          ? JSON.parse(ChartIcon)
                          : JSON.parse(ChartIcon1)
                      }
                    />
                  </OnTouch>
                  <ChartLegendlabelText>
                    Median of Category
                  </ChartLegendlabelText>
                </ChartLegendlabel>
              </OnTouch>
            </View>
          )}
        </ChartLegendGroup>
      </ChartLegend>
      {/* <ChartBackground> */}
      {/* <ChartItemLabelText>Item value avg</ChartItemLabelText> */}
      {/* <ChartTimeLabelText>Day</ChartTimeLabelText> */}
      <ChartFilterContainer>
        <OnTouch
          onPress={async () => {
            navigation.replace('JobDetail', {item: fullitem});
            // setIsLoading(true);
            // setCount('10');
            // setWronglength(null);
            // setDataOnCat([]);
            // setDataOnItem([]);
            // await getDaily();
          }}>
          {count === '10' ? (
            <Text style={{color: 'grey', fontSize: 10}}>10 Days </Text>
          ) : (
            <ChartFilterText>10 Days</ChartFilterText>
          )}
        </OnTouch>
        <ChartTimeLabelTextDivider />
        <OnTouch
          onPress={async () => {
            setIsLoading(true);
            setCount('30');
            setDataOnCat([]);
            setDataOnItem([]);
            await getDaily();
          }}>
          {count === '30' ? (
            <Text style={{color: 'grey', fontSize: 10}}>1 mon</Text>
          ) : (
            <ChartFilterText>1 mon</ChartFilterText>
          )}
        </OnTouch>
        <ChartTimeLabelTextDivider />
        <OnTouch
          onPress={async () => {
            setIsLoading(true);
            setCount('180');
            setDataOnCat([]);
            setWronglength(null);
            setDataOnItem([]);
            await getDaily();
          }}>
          {count === '180' ? (
            <Text style={{color: 'grey', fontSize: 10}}>6 mon</Text>
          ) : (
            <ChartFilterText>6 mon</ChartFilterText>
          )}
        </OnTouch>
        <ChartTimeLabelTextDivider />
        <OnTouch
          onPress={async () => {
            setIsLoading(true);
            setCount('365');
            setWronglength(null);
            setDataOnCat([]);
            setDataOnItem([]);
            await getDaily();
          }}>
          {count === '365' ? (
            <Text style={{color: 'grey', fontSize: 10}}>1 year</Text>
          ) : (
            <ChartFilterText>1 year</ChartFilterText>
          )}
        </OnTouch>
        <ChartTimeLabelTextDivider />
        <OnTouch
          onPress={async () => {
            setIsLoading(true);
            setWronglength(null);
            setDataOnCat([]);
            setDataOnItem([]);
            setCount('400');
            getDaily();
          }}>
          {count === '400' ? (
            <Text style={{color: 'grey', fontSize: 10}}>max</Text>
          ) : (
            <ChartFilterText>max</ChartFilterText>
          )}
        </OnTouch>
      </ChartFilterContainer>
      {/* {isLoading ? (
        <Loading size={18} color="#1cc910" />
      ) : (
        <> */}
      {parseInt(count) < itemLength ? (
        <LineChart
          // bezier
          withHorizontalLabels={true}
          withVerticalLabels={true}
          data={ChartData}
          width={getWidth * 0.9}
          height={200}
          withInnerLines={false}
          withOuterLines={true}
          chartConfig={chartConfig}
          style={chartStyle}
          bezier
        />
      ) : (
        // <Loading size={18} color="#1cc910" />
        <Text
          style={{
            color: '#000',
            margin: 20,
            fontSize: 14,
            textAlign: 'center',
          }}>
          Check back on this time range later after more data is collected for
          you
        </Text>
        // )}
        // </>
      )}
      {/* </ChartBackground> */}
    </>
  );
};

export default Chart;
