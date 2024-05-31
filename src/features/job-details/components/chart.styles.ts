import styled from 'styled-components/native';

export const ChartBackground = styled.ImageBackground.attrs({
  source: require('../../../../assets/outer-lines.png'),
  resizeMode: 'contain',
})`
  width: 100%;
  height: 140px;
`;
export const ChartLegend = styled.View`
  flex-direction: column;
  width: 100%;
`;
export const ChartLegendGroup = styled.View`
  flex-direction: row;
  margin-bottom: ${props => props.theme.space[2]};
  width: 80%;
`;
export const ChartLegendlabel = styled.View`
  flex-direction: row;
  align-items: center;
  flex: 1;
`;
export const ChartLegendlabelTextActive = styled.Text`
  color: ${props => props.theme.colors.text.tertiary};
  font-size: ${props => props.theme.fontSizes.subCaption};
`;
export const ChartLegendlabelText = styled.Text`
  color: ${props => props.theme.colors.text.primary};
  font-size: ${props => props.theme.fontSizes.subCaption};
`;
export const ChartIconImage = styled.Image`
  width: 12px;
  height: 12px;
  margin-right: ${props => props.theme.space[2]};
`;
export const ChartItemLabelText = styled.Text`
  transform: rotate(-90deg);
  position: absolute;
  bottom: 65px;
  left: -23px;
  color: ${props => props.theme.colors.text.quinary};
  font-size: 9px;
`;
export const ChartTimeLabelText = styled.Text`
  position: absolute;
  bottom: 18px;
  left: 20px;
  color: ${props => props.theme.colors.text.quinary};
  font-size: 9px;
`;
export const ChartFilterContainer = styled.View`
  flex-direction: column;
  width: 40px;
  position: absolute;
  right: 0;
  bottom: 32px;
  z-index: 1;
  padding: ${props => props.theme.space[1]}
  background-color: ${props => props.theme.colors.bg.quinary};
`;
export const OnTouch = styled.TouchableOpacity`
  align-items: center;
  padding: 4px;
`;
export const ChartFilterText = styled.Text`
  color: ${props => props.theme.colors.text.inverse}
  text-align: center;
  font-size: 9px;
`;
export const ChartTimeLabelTextDivider = styled.View`
  border-bottom-width: 1px;
  border-bottom-color: #eee;
  width: 100%;
  height: 1px;
`;
