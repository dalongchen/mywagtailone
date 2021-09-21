<template>
<div>
  <!-- flag === '1'为公告利好table 页面菜单stock  -->
  <div v-if="son_pro.flag === '1'">
    <el-table   
      :data="son_pro.data"
      style="width: 100%"
      @selection-change="handleSelectionChange"
      tooltip-effect="dark"
      ref="multipleTable"
    >
      <el-table-column type="selection" width="55"> </el-table-column>
      <el-table-column prop="da" label="日期" sortable></el-table-column>
      <el-table-column prop="xd_2102" label="代码" sortable></el-table-column>
      <el-table-column prop="xd_2103" label="名字" sortable width="100"></el-table-column>
      <el-table-column prop="close" label="收盘价"></el-table-column>
      <el-table-column prop="range_up" label="幅度"></el-table-column>
      <el-table-column label="买卖">
        <template #default="scope">
          <el-select v-model="scope.row.xd_2109">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            >
            </el-option>
          </el-select>
        </template>
      </el-table-column>  
      <el-table-column label="价格" width="160">
        <template #default="scope">
          <el-input-number  v-model= "scope.row.xd_2127" :step="0.01" size="mini" :min="0.01" 
          controls-position="right" @change= "handle_change( scope.row, 'range_up')"></el-input-number>
        </template>
      </el-table-column>  
      <el-table-column label="数量" width="160">
        <template #default="scope">
          <el-input-number  v-model= "scope.row.xd_2126" :step="100" size="mini" :min="1"
          controls-position="right" @change= "handle_change(scope.row, '')"></el-input-number>
        </template>
      </el-table-column>  
      <el-table-column prop="mor" label="金额"></el-table-column> 
      <el-table-column prop="xd_2105" label="状态"></el-table-column>  
    </el-table>
    <el-descriptions :column="4">
      <el-descriptions-item  width="100px" align="center">
        <el-button type="primary" icon="el-icon-edit" @click="handle_submit('edit')" 
        :disabled="disable"></el-button>
      </el-descriptions-item>
      <el-descriptions-item width="100px" align="center">
        <el-button type="primary" icon="el-icon-delete"  @click="handle_submit('delete')" 
        :disabled="disable2"></el-button>
      </el-descriptions-item>
      <el-descriptions-item width="100px" align="center" label="总买">{{buy}}</el-descriptions-item>
      <el-descriptions-item width="100px" align="center" label="总卖">{{sell}}</el-descriptions-item>
    </el-descriptions>
  </div>
</div>
</template>
 
<script lang="">
import { defineComponent } from "vue";
import axios from "axios";

export default defineComponent({
  name: "TableTrade",
  props: {//接收父组件传递过来的参数
    son_props:Object,
    map:{
      // type:String
    },
  },
  data() {
    return {
      disable: true,
      disable2: true,
      son_pro: Object,
      multiple: [],
      multipleSelection: [],
      buy: 0,
      sell: 0,
      options: [
        {
          value: '买入',
          label: '买入',
        },
        {
          value: '卖出',
          label: '卖出',
        },
      ],
    } 
  },
  mounted() {
    // console.log(son_props.data)
  },
  watch: {   //监听值变化:map值
		son_props:function (son_props) {
      for (let d of son_props.data) {
        d["xd_2126"] = Number(d["xd_2126"])
        d["xd_2127"] = Number(d["xd_2127"])
        d["mor"] = (d["xd_2127"]*d["xd_2126"]).toFixed(1)
        d["range_up"] = (((d["xd_2127"]-d["close"])/d["close"])*100).toFixed(2)
        if (d["xd_2109"] === "买入"){
          this.buy += Number(d["mor"])
        }
        if (d["xd_2109"] === "卖出"){
          this.sell += Number(d["mor"])
        }
      }
      this.son_pro = this.son_props
      // son_props.data.push({"close":"总买:", "xd_2127":buy, "xd_2126":"总卖:", "mor":sell})
		}
	},
  computed: {
    // handle_change(v) {
    //   console.log(v)
    //   // return 3
    // }
  },
  methods: {
    handleSelectionChange(val) {
      this.multipleSelection = val
      // console.log(val.length)
      if (this.multipleSelection.length === 0){ this.disable2 = true }else{ this.disable2 = false }
      if (val.length === 0){ this.disable = true }else{
        this.multiple= val.map((i) => { return (i.xd_2105 === "" ? ({"xd_2102": i.xd_2102,
        "xd_2126": i.xd_2126,"xd_2127": i.xd_2127,"xd_2109": i.xd_2109,}) : "")})
        this.multiple = this.multiple.filter(item => item!=="" )
        if (this.multiple.length!==0 ){ this.disable = false }
      }
    },
    handle_change(row, f) {
      let r = row.mor
      if (f === "range_up"){
        row.range_up = (((row.xd_2127-row.close)/row.close)*100).toFixed(2)
      }
      row.mor = (row.xd_2127*row.xd_2126).toFixed(1)
      // console.log(row.mor - r)
      if (row.xd_2109 === "买入"){
          this.buy = this.buy + Number(row.mor - r)
        }
      if (row.xd_2109 === "卖出"){
        this.sell = this.sell + Number(row.mor - r)
      }
      
    },
    handle_submit(s) {
      this.disable = true 
      this.disable2 = true 
      if (s === "delete"){
        this.multiple = this.multipleSelection.map(item => ({"xd_2102": item.xd_2102}))
      }
      axios.get("http://127.0.0.1:8000/datatables/easy_trade/", {
        params: { s: s, d: JSON.stringify(this.multiple)}
      }).then(response => {
        if (response.status === 200){
          if (s === "delete"){
            for (let mul of this.multipleSelection) {
              if (mul.xd_2109 === "买入"){
                  this.buy = this.buy - Number(mul.mor)
              }
              if (mul.xd_2109 === "卖出"){
                this.sell = this.sell - Number(mul.mor)
              }
              this.son_pro.data.splice(this.son_pro.data.findIndex(item => item.xd_2102 === mul.xd_2102),1)
            }
            this.$refs.multipleTable.clearSelection();
            this.multipleSelection = []
          }else if (s === "edit"){
            this.$refs.multipleTable.clearSelection();
            this.multipleSelection = []
            this.multiple = []
          }
        }
      }).catch(error => {
        console.log(error);
      });
    },
   
  },
});
</script>
<style lang="scss">
.el-checkbox__inner {
  border-color: #60f715;
} 
</style>