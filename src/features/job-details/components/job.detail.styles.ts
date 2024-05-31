import {Dimensions, ScrollView} from 'react-native';

import styled from 'styled-components/native';

export const getWidth = Dimensions.get('window').width;

export const OnScroll = styled(ScrollView).attrs({
  contentContainerStyle: {paddingBottom: 80},
})`
  height: 100%;
  background: rgba(255, 255, 255, 0.87);
  padding-bottom: 500px;
`;
export const ItemDetailContainer = styled.View`
  height: 100%;
`;
export const ItemDetailContainerPadding = styled.View`
  padding-left: ${props => props.theme.space[6]};
  padding-right: ${props => props.theme.space[6]};
`;
export const ItemTitleContainer = styled.View`
  flex-direction: row;
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
  border-bottom-width: 1px;
  border-bottom-color: ${props => props.theme.colors.ui.quinary}
`;
export const ItemTier = styled.View`
  width: 33.333%;
  flex-direction: row;
  align-items: center;
`;
export const TierImage = styled.Image`
  width: 30px;
  height: 30px;
`;
export const TierLevelIcon = styled.View`
  width: 30px;
  height: 30px;
`;
export const TierLevelText = styled.Text`
  font-weight: 500;
  position: absolute;
  left: 0;
  right: 0;
  top: 5px;
  bottom: 0;
  text-align: center;
  z-index: 1;
  color: #096c5c;
  font-size: 18px;
`;
export const UserTierText = styled.Text`
  background-color: #3CB474;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.subText}
`;

export const ItemGoBack = styled.View`
  width: 33.333%;
  flex-direction: row;
  justify-content: flex-end;
`;
export const ItemGoBackImage = styled.Image`
  width: 24px;
  height: 24px;
  align-self: flex-end;
`;
export const OnTouch = styled.TouchableOpacity``;

export const ItemDescription = styled.View`
  flex-direction: row;
  border-bottom-width: 1px;
  border-right-width: 1px;
  border-bottom-color: ${props => props.theme.colors.ui.quinary};
  border-right-color: ${props => props.theme.colors.ui.quinary};
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
  margin-bottom: ${props => props.theme.space[4]}
`;
export const ItemDescriptionImage = styled.Image`
  width: 60px;
  height: 68px;
  border-radius: ${props => props.theme.space[2]};
`;
export const OnTouchEditImage = styled.TouchableOpacity`
  width: 26px;
  height: 30px;
  position: absolute;
  top: 32px;
  left: 19px;
  z-index: 1;
`;
export const ItemEditImage = styled.Image`
  width: 26px;
  height: 30px;
`;
export const ItemDescriptionDetail = styled.View`
  flex-direction: column;
  flex: 1;
  padding-left: ${props => props.theme.space[3]}
  padding-right: ${props => props.theme.space[3]}
`;
export const ItemDescriptionDetailHeader = styled.View`
  flex-direction: row;
  justify-content: space-between;
  padding-bottom: ${props => props.theme.space[2]};
  align-items: center;
`;
export const ItemAddedDateText = styled.Text`
  font-size: ${props => props.theme.fontSizes.subCaption};
  color: ${props => props.theme.colors.text.secondary};
  font-weight: ${props => props.theme.fontWeights.medium};
`;
export const ItemDescriptionEditImage = styled.Image`
  width: 16px;
  height: 16px;
`;
export const ItemDescriptionText = styled.Text`
  font-size: ${props => props.theme.fontSizes.subCaption};
  color: ${props => props.theme.colors.text.primary};
`;
export const ItemTierDivider = styled.View`
  border-bottom-width: 1px;
  border-bottom-color: ${props => props.theme.colors.ui.quinary};
  height: 1px;
  flex: 1;
`;
export const ChartContainer = styled.View`
  margin-bottom: ${props => props.theme.space[4]};
`;
export const ScrollViwWrapper = styled.View`
  height: 80px;
`;
export const OnScrollHorizontal = styled.ScrollView``;
export const ItemUserTier = styled.View`
  flex-direction: row;
  justify-content: center;
  align-items: center;
  background: #8ecfad;
  height: 65px;
  width: ${getWidth / 2 - 34}px;
  border-radius: ${props => props.theme.space[3]};
  margin-right: ${props => props.theme.space[3]}
  margin-left: ${props => props.theme.space[1]}
`;
export const ItemUserTierImage = styled.Image`
  width: 56px;
  height: 64px;
  border-radius: ${props => props.theme.space[2]};
  margin-top: -5px;
  margin-left: -3px;
`;
export const ItemUserTierTitleText = styled.Text`
  font-size: ${props => props.theme.fontSizes.subText};
  color: ${props => props.theme.colors.text.inverse};
  font-weight: ${props => props.theme.fontWeights.bold};
  text-decoration: underline;
  text-decoration-color: ${props => props.theme.colors.ui.quaternary}
  padding: ${props => props.theme.space[2]};
  flex: 1;
`;
export const ItemUserTierVariant = styled.View`
  flex-direction: row;
  justify-content: center;
  align-items: center;
  background: #EEC689;
  height: 65px;
  width: ${getWidth / 2 - 34}px;
  border-radius: ${props => props.theme.space[3]};
  margin-right: ${props => props.theme.space[3]}
  margin-left: ${props => props.theme.space[1]}
`;
