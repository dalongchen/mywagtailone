<template>
<div>
  <!-- flag === '1'为公告利好table 页面菜单stock  -->
  <div v-if="son_props.flag === '1'">
    <el-table   
      :data="son_props.data"
      style="width: 100%"
    >
      <el-table-column prop="notice_date" label="日期" sortable width="180"></el-table-column>
      <el-table-column prop="dis_time" label="公告" sortable width="200"></el-table-column>
      <el-table-column prop="col_type" label="类型" width="180"></el-table-column>
      <el-table-column prop="title" label="标题"></el-table-column>  
    </el-table>
  </div>
  <!-- flag === '2'为列标题传入的公用表,data数据为2维列表list 列表字典  -->
  <div v-else-if="son_props.flag === '2'">
    <el-tag type="success" class="tag_right">{{son_props.caption}}</el-tag>
    <el-table
      :data="son_props.data"
      style="width: 100%"
      empty-text="null"
    >
      <el-table-column
          :width="son_props.width" 
          v-for=" i in son_props.th"
          :key="i.id"
          :property="i.prop"
          :label="i.propName">
      </el-table-column> 
    </el-table>
  </div>
  <!-- flag === '3'为不需要表头,data为二维数组  -->
  <div v-else-if="son_props.flag === '3'">
    <el-tag type="success" class="tag_right">{{son_props.caption}}</el-tag>
    <el-table
      :data="son_props.data"
      style="width: 100%"
      empty-text="null"
      :show-header="false"
    >
      <el-table-column
      v-for=" i in son_props.th"
      :key="i.id"
      :property="i.prop"
      :show-overflow-tooltip="i.t"
      :width="i.width" >
      </el-table-column>
    </el-table>
  </div>
  <!-- flag === '30'为不需要表头,data为二维数组 带点击 -->
  <div v-else-if="son_props.flag === '30'">
    <el-tag type="success" class="tag_right">{{son_props.caption}}</el-tag>
    <el-table
      :data="son_props.data"
      style="width: 100%"
      empty-text="null"
      :show-header="false"
      @cell-click="handle"
    >
      <el-table-column
      v-for=" i in son_props.th"
      :key="i.id"
      :property="i.prop"
      :show-overflow-tooltip="i.t"
      :width="i.width" >
      </el-table-column>
    </el-table>
  </div>
  <!-- flag === '4'循环表 字典的data -->
  <div v-else-if="son_props.flag === '4'">
    <div v-for=" i in son_props.data" :key="i.id">
      <el-tag type="success" class="tag_right">{{i.rq + son_props.caption}}</el-tag>
      <el-table
        :data="i.sdgd"
        style="width: 100%"
        empty-text="null"
        :show-header="false"
      >
        <el-table-column
        v-for=" i in son_props.th"
        :key="i.id"
        :property="i.prop"
        :width="i.width" >
        </el-table-column>
      </el-table>
    </div>
  </div>
  <!-- flag === '5'循环表 列表的data -->
  <div v-else-if="son_props.flag === '5'">
    <div v-for=" i in son_props.data" :key="i.id">
      <el-tag type="success" class="tag_right">{{son_props.caption}}</el-tag>
      <el-table
        :data="i"
        style="width: 100%"
        empty-text="null"
        :show-header="false"
      >
        <el-table-column
        v-for=" i in son_props.th"
        :key="i.id"
        :property="i.prop"
        :width="i.width" >
        </el-table-column>
      </el-table>
    </div>
  </div>
</div>
</template>
 
<script lang="">
import { defineComponent } from "vue";

export default defineComponent({
  name: "ElTableSon",
  props: {
    son_props:Object,
  },
  data() {
    return {
      // a,
      // a:[
      //             "0",
      //             "1",
      //             "2",
      //           ]
    } 
  },
  computed: {
    pu() {
      return Array.from(Array(this.son_props.data[0].length), (v,k) =>k +"")
    }
  },
  methods: {
    handle(row){
      console.log(row["4"].startsWith('http'))
      if (row["4"].startsWith('http')) {
        window.open(row["4"])
      }
    }
  },
});
</script>
<style lang="scss">
.tag_right{float:left;width:300px} 
</style>